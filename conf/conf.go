package conf

import "os"

var ENV string = os.Getenv("STAGE")
