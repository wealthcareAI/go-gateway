package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
	"google.golang.org/genai"
)

/*
ENV
----
MCP_SERVER_URL   (required) e.g. https://your-mcp.example.com/mcp
DATABASE_URL     (required) e.g. postgres://user:pass@host:5432/dbname?sslmode=disable
GEMINI_API_KEY   or GOOGLE_API_KEY (optional; SDK also reads env automatically)
GEMINI_MODEL     (default: gemini-2.0-flash)
ADDR             (default: :8080)
*/

type Config struct {
	MCPServerURL string
	DBURL        string
	APIKey       string // GEMINI_API_KEY or GOOGLE_API_KEY
	Model        string
	Addr         string
}

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
func mustConfig() Config {
	cfg := Config{
		MCPServerURL: os.Getenv("MCP_SERVER_URL"),
		DBURL:        os.Getenv("DATABASE_URL"),
		APIKey:       getenv("GEMINI_API_KEY", getenv("GOOGLE_API_KEY", "")),
		Model:        getenv("GEMINI_MODEL", "gemini-2.5-flash"),
		Addr:         getenv("ADDR", ":4000"),
	}
	if cfg.MCPServerURL == "" {
		log.Fatal("set MCP_SERVER_URL")
	}
	if cfg.DBURL == "" {
		log.Fatal("set DATABASE_URL")
	}
	return cfg
}

/* ---------- Postgres ---------- */

type Store struct{ pool *pgxpool.Pool }

func (s *Store) Migrate(ctx context.Context) error {
	sql := `
CREATE TABLE IF NOT EXISTS chat_sessions (
  id UUID PRIMARY KEY,
  title TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_messages (
  id UUID PRIMARY KEY,
  session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL,        -- 'user' | 'assistant' | 'tool' | 'system'
  content TEXT,              -- plain text message content
  raw JSONB,                 -- optional structured payload
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created ON chat_messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS tool_invocations (
  id UUID PRIMARY KEY,
  session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  step_index INT NOT NULL,
  tool_name TEXT NOT NULL,
  args JSONB,
  result_text TEXT,
  raw JSONB,
  error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_tool_invocations_session_created ON tool_invocations(session_id, created_at);
`
	_, err := s.pool.Exec(ctx, sql)
	return err
}

func (s *Store) CreateSession(ctx context.Context, title string) (uuid.UUID, error) {
	id := uuid.New()
	_, err := s.pool.Exec(ctx, `INSERT INTO chat_sessions(id, title) VALUES ($1,$2)`, id, title)
	return id, err
}

func (s *Store) TouchSession(ctx context.Context, id uuid.UUID) {
	_, _ = s.pool.Exec(ctx, `UPDATE chat_sessions SET updated_at = now() WHERE id=$1`, id)
}

func (s *Store) SaveMessage(ctx context.Context, sessionID uuid.UUID, role, content string, raw any) error {
	id := uuid.New()
	var rawJSON []byte
	if raw != nil {
		rawJSON, _ = json.Marshal(raw)
	}
	_, err := s.pool.Exec(ctx, `
INSERT INTO chat_messages(id, session_id, role, content, raw) VALUES ($1,$2,$3,$4,$5)
`, id, sessionID, role, content, rawJSON)
	return err
}

func (s *Store) SaveToolInvocation(ctx context.Context, sessionID uuid.UUID, step int, name string, args any, resultText string, raw any, errStr string) error {
	id := uuid.New()
	argsJSON, _ := json.Marshal(args)
	var rawJSON []byte
	if raw != nil {
		rawJSON, _ = json.Marshal(raw)
	}
	_, err := s.pool.Exec(ctx, `
INSERT INTO tool_invocations(id, session_id, step_index, tool_name, args, result_text, raw, error)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`, id, sessionID, step, name, argsJSON, resultText, rawJSON, errStr)
	return err
}

type DBMessage struct {
	Role    string
	Content string
	Created time.Time
}

// returns prior user/assistant messages as genai contents (ordered oldest→newest)
func (s *Store) LoadHistoryAsContents(ctx context.Context, sessionID uuid.UUID, limit int) ([]*genai.Content, error) {
	if limit <= 0 || limit > 1000 {
		limit = 50
	}
	rows, err := s.pool.Query(ctx, `
SELECT role, content, created_at
FROM chat_messages
WHERE session_id=$1 AND role IN ('user','assistant')
ORDER BY created_at ASC
LIMIT $2`, sessionID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []*genai.Content
	for rows.Next() {
		var m DBMessage
		if err := rows.Scan(&m.Role, &m.Content, &m.Created); err != nil {
			return nil, err
		}
		role := genai.Role("user")
		if m.Role == "assistant" {
			role = genai.Role("model")
		}
		out = append(out, genai.NewContentFromText(m.Content, role))
	}
	return out, rows.Err()
}

/* ---------- SamplingHandler (for server-initiated sampling) ---------- */

type ctxKey string

const (
	ctxPromptKey  ctxKey = "sysPrompt"
	ctxModelKey   ctxKey = "model"
	ctxTempKey    ctxKey = "temperature"
	ctxMaxTokKey  ctxKey = "maxTokens"
	ctxStopSeqKey ctxKey = "stopSeq"
	ctxSessionKey ctxKey = "sessionID" // for saving messages if desired
)

func ctxGet[T any](ctx context.Context, k ctxKey, def T) T {
	if v := ctx.Value(k); v != nil {
		if vv, ok := v.(T); ok {
			return vv
		}
	}
	return def
}

type GeminiSampler struct {
	Client       *genai.Client
	DefaultModel string
}

func (g *GeminiSampler) CreateMessage(ctx context.Context, req mcp.CreateMessageRequest) (*mcp.CreateMessageResult, error) {
	modelName := ctxGet[string](ctx, ctxModelKey, g.DefaultModel)
	cfg := &genai.GenerateContentConfig{}
	if t := ctxGet[float32](ctx, ctxTempKey, 0); t != 0 {
		cfg.Temperature = &t
	}
	if mt := ctxGet[int32](ctx, ctxMaxTokKey, 0); mt > 0 {
		cfg.MaxOutputTokens = mt
	}
	if stop := ctxGet[[]string](ctx, ctxStopSeqKey, nil); len(stop) > 0 {
		cfg.StopSequences = stop
	}

	// System instruction = request prompt + server SystemPrompt
	var sysParts []*genai.Part
	if p := strings.TrimSpace(ctxGet[string](ctx, ctxPromptKey, "")); p != "" {
		sysParts = append(sysParts, &genai.Part{Text: p})
	}
	if sp := strings.TrimSpace(req.CreateMessageParams.SystemPrompt); sp != "" {
		sysParts = append(sysParts, &genai.Part{Text: sp})
	}
	if len(sysParts) > 0 {
		cfg.SystemInstruction = &genai.Content{Parts: sysParts}
	}

	// MCP messages -> []*genai.Content
	var contents []*genai.Content
	for _, m := range req.CreateMessageParams.Messages {
		if tc, ok := mcp.AsTextContent(m.Content); ok {
			role := genai.Role("user")
			if m.Role == mcp.RoleAssistant {
				role = genai.Role("model")
			}
			contents = append(contents, genai.NewContentFromText(tc.Text, role))
		}
	}

	resp, err := g.Client.Models.GenerateContent(ctx, modelName, contents, cfg)
	if err != nil {
		return nil, err
	}

	return &mcp.CreateMessageResult{
		SamplingMessage: mcp.SamplingMessage{
			Role:    mcp.RoleAssistant,
			Content: mcp.NewTextContent(resp.Text()),
		},
		Model:      modelName,
		StopReason: "end_turn",
	}, nil
}

/* ---------- Agent (function-calling loop) ---------- */

type StepTrace struct {
	CallName string              `json:"call_name,omitempty"`
	Args     map[string]any      `json:"args,omitempty"`
	Text     string              `json:"text,omitempty"`  // extracted tool text
	Error    string              `json:"error,omitempty"` // tool error (if any)
	Raw      *mcp.CallToolResult `json:"raw,omitempty"`
	Final    bool                `json:"final,omitempty"`
}

func buildFunctionDecls(tools []mcp.Tool) ([]*genai.FunctionDeclaration, map[string]mcp.Tool) {
	decls := make([]*genai.FunctionDeclaration, 0, len(tools))
	index := make(map[string]mcp.Tool, len(tools))
	for _, t := range tools {
		b, _ := json.Marshal(t)
		var m map[string]any
		_ = json.Unmarshal(b, &m)
		name, _ := m["name"].(string)
		if name == "" {
			continue
		}
		desc, _ := m["description"].(string)
		params := m["inputSchema"]
		fd := &genai.FunctionDeclaration{Name: name, Description: desc}
		if params != nil {
			fd.ParametersJsonSchema = params
		}
		decls = append(decls, fd)
		index[name] = t
	}
	return decls, index
}

// agentLoop: model proposes function calls; we execute them; we loop.
// history is injected (as prior contents) before userQ.
func agentLoop(
	ctx context.Context,
	gen *genai.Client,
	model string,
	mcpClient *mcpclient.Client,
	history []*genai.Content,
	userQ string,
	sysPrompt string,
	maxSteps int,
	forceTools bool,
	onTool func(step int, name string, args map[string]any, res *mcp.CallToolResult, err error),
) (finalText string, trace []StepTrace, err error) {

	// tools -> function declarations
	tl, err := mcpClient.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return "", nil, fmt.Errorf("list tools: %w", err)
	}
	decls, _ := buildFunctionDecls(tl.Tools)

	cfg := &genai.GenerateContentConfig{Tools: []*genai.Tool{{FunctionDeclarations: decls}}}
	if sysPrompt != "" {
		cfg.SystemInstruction = &genai.Content{Parts: []*genai.Part{{Text: sysPrompt}}}
	}
	if forceTools {
		cfg.ToolConfig = &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAny}}
	} else {
		cfg.ToolConfig = &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAuto}}
	}

	contents := make([]*genai.Content, 0, len(history)+1)
	contents = append(contents, history...)
	contents = append(contents, genai.NewContentFromText(userQ, genai.Role("user")))

	if maxSteps <= 0 || maxSteps > 16 {
		maxSteps = 8
	}

	for step := 0; step < maxSteps; step++ {
		resp, err := gen.Models.GenerateContent(ctx, model, contents, cfg)
		if err != nil {
			return "", trace, err
		}

		calls := resp.FunctionCalls()
		if len(calls) == 0 {
			// final answer
			final := resp.Text()
			trace = append(trace, StepTrace{Final: true, Text: final})
			return final, trace, nil
		}

		// Execute calls in order; append tool responses and continue
		for _, c := range calls {
			args := map[string]any{}
			for k, v := range c.Args {
				args[k] = v
			}

			res, callErr := mcpClient.CallTool(ctx, mcp.CallToolRequest{
				Params: mcp.CallToolParams{Name: c.Name, Arguments: args},
			})

			if onTool != nil {
				onTool(step, c.Name, args, res, callErr)
			}

			st := StepTrace{CallName: c.Name, Args: args, Raw: res}
			if callErr != nil {
				st.Error = callErr.Error()
				trace = append(trace, st)
				errPayload := map[string]any{"error": st.Error}
				contents = append(contents, genai.NewContentFromFunctionResponse(c.Name, errPayload, genai.Role("tool")))
				continue
			}
			st.Text = extractToolText(res)
			trace = append(trace, st)

			payload := map[string]any{"text": st.Text, "raw": res}
			contents = append(contents, genai.NewContentFromFunctionResponse(c.Name, payload, genai.Role("tool")))
		}
	}
	return "", trace, fmt.Errorf("maxSteps exceeded without final answer")
}

/* ---------- HTTP server ---------- */

type App struct {
	cfg Config
	db  *Store
	mcp *mcpclient.Client
	gen *genai.Client
}

func main() {
	cfg := mustConfig()
	ctx := context.Background()

	// Postgres
	pool, err := pgxpool.New(ctx, cfg.DBURL)
	if err != nil {
		log.Fatalf("db: %v", err)
	}
	defer pool.Close()
	store := &Store{pool: pool}
	if err := store.Migrate(ctx); err != nil {
		log.Fatalf("db migrate: %v", err)
	}

	// GenAI client
	var genClient *genai.Client
	if cfg.APIKey != "" {
		genClient, err = genai.NewClient(ctx, &genai.ClientConfig{APIKey: cfg.APIKey, Backend: genai.BackendGeminiAPI})
	} else {
		genClient, err = genai.NewClient(ctx, &genai.ClientConfig{Backend: genai.BackendGeminiAPI})
	}
	if err != nil {
		log.Fatalf("genai: %v", err)
	}
	// defer genClient.Close()

	// MCP client (Streamable HTTP) + sampling handler
	sampler := &GeminiSampler{Client: genClient, DefaultModel: cfg.Model}
	mcpc, err := mcpclient.NewStreamableHttpClient(cfg.MCPServerURL)
	if err != nil {
		log.Fatalf("mcp client: %v", err)
	}
	mcpclient.WithSamplingHandler(sampler)(mcpc)

	if err := mcpc.Start(ctx); err != nil {
		log.Fatalf("mcp start: %v", err)
	}
	defer mcpc.Close()
	if _, err := mcpc.Initialize(ctx, mcp.InitializeRequest{
		Params: mcp.InitializeParams{
			ProtocolVersion: mcp.LATEST_PROTOCOL_VERSION,
			ClientInfo:      mcp.Implementation{Name: "mcp-agent-db", Version: "0.1.0"},
			Capabilities:    mcp.ClientCapabilities{Sampling: &struct{}{}},
		},
	}); err != nil {
		log.Fatalf("mcp initialize: %v", err)
	}

	app := &App{cfg: cfg, db: store, mcp: mcpc, gen: genClient}

	r := chi.NewRouter()
	r.Post("/sessions", app.handleCreateSession)
	r.Get("/sessions/{id}/history", app.handleGetHistory)
	r.Get("/tools", app.handleListTools)
	r.Post("/query", app.handleQueryAgent)

	log.Printf("listening on %s", cfg.Addr)
	if err := http.ListenAndServe(cfg.Addr, r); err != nil {
		log.Fatal(err)
	}
}

/* ---- Sessions ---- */

type createSessionReq struct {
	Title string `json:"title,omitempty"`
}
type createSessionResp struct {
	SessionID string `json:"session_id"`
}

func (a *App) handleCreateSession(w http.ResponseWriter, r *http.Request) {
	var req createSessionReq
	_ = json.NewDecoder(r.Body).Decode(&req)
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	id, err := a.db.CreateSession(ctx, req.Title)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, createSessionResp{SessionID: id.String()})
}

func (a *App) handleGetHistory(w http.ResponseWriter, r *http.Request) {
	idStr := chi.URLParam(r, "id")
	// sid, err := uuid.Parse(idStr)
	// if err != nil {
	// 	http.Error(w, "bad session id", http.StatusBadRequest)
	// 	return
	// }
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit <= 0 || limit > 1000 {
		limit = 50
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	rows, err := a.db.pool.Query(ctx, `
SELECT role, content, created_at
FROM chat_messages
WHERE session_id=$1
ORDER BY created_at ASC
LIMIT $2`, idStr, limit)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	type msg struct {
		Role, Content string
		CreatedAt     time.Time
	}
	var msgs []msg
	for rows.Next() {
		var m msg
		if err := rows.Scan(&m.Role, &m.Content, &m.CreatedAt); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		msgs = append(msgs, m)
	}
	writeJSON(w, http.StatusOK, msgs)
}

/* ---- Tools ---- */

func (a *App) handleListTools(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()
	tl, err := a.mcp.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, tl.Tools)
}

/* ---- Agent (query) ---- */

type queryReq struct {
	SessionID   string   `json:"session_id,omitempty"`
	Q           string   `json:"q"`
	Prompt      string   `json:"prompt,omitempty"` // system instruction
	Model       string   `json:"model,omitempty"`
	Temperature float32  `json:"temperature,omitempty"`
	MaxTokens   int32    `json:"max_tokens,omitempty"`
	Stop        []string `json:"stop,omitempty"`
	MaxSteps    int      `json:"max_steps,omitempty"`
	ForceTools  bool     `json:"force_tools,omitempty"`
	TimeoutMS   int      `json:"timeout_ms,omitempty"`
}

type queryResp struct {
	SessionID string      `json:"session_id"`
	Final     string      `json:"final"`
	Trace     []StepTrace `json:"trace"`
	Err       string      `json:"err,omitempty"`
}

func (a *App) handleQueryAgent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var req queryReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Q) == "" {
		http.Error(w, `missing "q"`, http.StatusBadRequest)
		return
	}

	// Ensure session
	var sid uuid.UUID
	var err error
	if req.SessionID == "" {
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		sid, err = a.db.CreateSession(ctx, "")
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
	} else {
		sid, err = uuid.Parse(req.SessionID)
		if err != nil {
			http.Error(w, "bad session_id", http.StatusBadRequest)
			return
		}
	}
	a.db.TouchSession(r.Context(), sid)

	// Save user message
	_ = a.db.SaveMessage(r.Context(), sid, "user", req.Q, nil)

	// Build sampling context for any server-initiated sampling during tool runs
	ctx := r.Context()
	if req.TimeoutMS > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(req.TimeoutMS)*time.Millisecond)
		defer cancel()
	}
	if req.Prompt != "" {
		ctx = context.WithValue(ctx, ctxPromptKey, req.Prompt)
	}
	if req.Model != "" {
		ctx = context.WithValue(ctx, ctxModelKey, req.Model)
	}
	if req.Temperature != 0 {
		ctx = context.WithValue(ctx, ctxTempKey, req.Temperature)
	}
	if req.MaxTokens > 0 {
		ctx = context.WithValue(ctx, ctxMaxTokKey, req.MaxTokens)
	}
	if len(req.Stop) > 0 {
		ctx = context.WithValue(ctx, ctxStopSeqKey, req.Stop)
	}
	ctx = context.WithValue(ctx, ctxSessionKey, sid)

	// Load prior user/assistant history as context
	history, err := a.db.LoadHistoryAsContents(ctx, sid, 100)
	if err != nil {
		http.Error(w, "load history: "+err.Error(), 500)
		return
	}

	modelName := firstNonEmpty(req.Model, a.cfg.Model)
	stepIdx := 0
	onTool := func(step int, name string, args map[string]any, res *mcp.CallToolResult, callErr error) {
		stepIdx++
		var errStr string
		var txt string
		if callErr != nil {
			errStr = callErr.Error()
		}
		if res != nil {
			txt = extractToolText(res)
		}
		_ = a.db.SaveToolInvocation(context.Background(), sid, stepIdx, name, args, txt, res, errStr)
		// Optional: persist tool outputs in chat timeline as well
		if txt != "" {
			_ = a.db.SaveMessage(context.Background(), sid, "tool", txt, map[string]any{"tool": name, "args": args})
		}
	}

	final, trace, loopErr := agentLoop(ctx, a.gen, modelName, a.mcp, history, req.Q, req.Prompt, req.MaxSteps, req.ForceTools, onTool)
	if loopErr != nil {
		writeJSON(w, http.StatusBadGateway, queryResp{SessionID: sid.String(), Trace: trace, Err: loopErr.Error()})
		return
	}

	// Save assistant final reply
	_ = a.db.SaveMessage(context.Background(), sid, "assistant", final, nil)
	writeJSON(w, http.StatusOK, queryResp{SessionID: sid.String(), Final: final, Trace: trace})
}

/* ---------- helpers ---------- */

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func extractToolText(res *mcp.CallToolResult) string {
	if res == nil {
		return ""
	}
	var out []string
	for _, c := range res.Content {
		if tc, ok := mcp.AsTextContent(c); ok && strings.TrimSpace(tc.Text) != "" {
			out = append(out, tc.Text)
		}
	}
	return strings.Join(out, "\n")
}
