record:
    go run ./cmd/record

pull-recordings:
    go run ./cmd/contrib pull

gen-assignment name:
    go run ./cmd/contrib gen-assignment {{name}}

gen-token name ttl="168h":
    go run ./cmd/contrib gen-token {{name}} {{ttl}}

add-wanted name:
    go run ./cmd/contrib add-wanted {{name}}

list-progress name="":
    go run ./cmd/contrib list-progress {{name}}

eval *args:
    uv run python cmd/eval/main.py {{args}}

train:
    uv run python cmd/train/train_loop.py

deploy-worker:
    npm --prefix worker run deploy

generate-audio csv:
    uv run python cmd/audio_gen/generate_audio.py {{csv}} \
    --references data/speaker-clips/jp --out-root out --device mps