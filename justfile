record:
    go run ./cmd/record

pull-recordings:
    go run ./cmd/contrib pull

gen-assignment name:
    go run ./cmd/contrib gen-assignment {{name}}

gen-token name ttl="168h":
    go run ./cmd/contrib gen-token {{name}} {{ttl}}

list-progress name="":
    go run ./cmd/contrib list-progress {{name}}
