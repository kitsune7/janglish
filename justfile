record:
    go run ./cmd/record

pull-recordings:
    go run ./cmd/contrib pull

gen-assignment name +ids:
    go run ./cmd/contrib gen-assignment {{name}} {{ids}}

gen-token name ttl="168h":
    go run ./cmd/contrib gen-token {{name}} {{ttl}}

list-progress name="":
    go run ./cmd/contrib list-progress {{name}}
