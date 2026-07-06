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

predict checkpoint target *args:
    uv run python cmd/train/predict.py {{checkpoint}} {{target}} {{args}}

deploy-worker:
    npm --prefix worker run deploy

audio-gen csv workers="1" synthetic_folder="":
    PYTHONWARNINGS="ignore:An output with one or more elements was resized:UserWarning" \
    uv run python cmd/audio_gen/generate_audio.py {{csv}} --has-header \
    --references data/speaker-clips/en data/speaker-clips/jp --out-root data --device mps --seed 42 \
    --workers {{workers}}{{ if synthetic_folder == "" { "" } else { " --synthetic-folder " + synthetic_folder } }}