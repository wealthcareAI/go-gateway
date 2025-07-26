package main

import (
	"net/http"
	"time"
	"walthcareai/conf"
	"walthcareai/constants"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

func main() {
	r := chi.NewRouter()
	r.Use(middleware.Recoverer)
	r.Use(secureMiddleware)

	r.Get("/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("health ok"))
	})
	r.Post("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("health ok"))
	})

	startServer := &http.Server{
		Addr:              ":9999",
		ReadHeaderTimeout: 3 * time.Second,
		Handler:           r,
		IdleTimeout:       65 * time.Second,
	}

	startServer.ListenAndServe()
}

func secureMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {

		if conf.ENV != constants.ENV_LOCAL {
			w.Header().Set("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload")
			w.Header().Set("Content-Security-Policy", "default-src 'self'")
			w.Header().Set("X-Frame-Options", "DENY")
		}
		next.ServeHTTP(w, r)
	})
}
