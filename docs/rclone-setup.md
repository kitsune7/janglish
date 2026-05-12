# rclone + Cloudflare R2 setup

One-time setup so `cmd/contrib` can read from and write to the `janglish-recordings` R2 bucket.

## Prerequisites

- `brew install rclone`
- An R2 API token with **Object Read & Write** scoped to the `janglish-recordings` bucket. From the Cloudflare dashboard → R2 → **Manage R2 API Tokens** → **Create API Token**. Save the Access Key ID, Secret Access Key, and the S3 endpoint URL (looks like `https://<account-id>.r2.cloudflarestorage.com`).

## Configure the remote

Run:

```
rclone config
```

Answer the prompts:

- `n` — new remote
- name: `r2`
- storage: `s3` (Amazon S3 Compliant Storage Providers)
- provider: `Cloudflare`
- env_auth: `false`
- access_key_id: *paste from the token*
- secret_access_key: *paste from the token*
- region: `auto`
- endpoint: *paste the `https://<account-id>.r2.cloudflarestorage.com` URL*
- location_constraint, ACL, etc.: leave blank
- Edit advanced config: `n`
- Confirm: `y`, then `q` to quit

Then add one extra line (the R2 API token is bucket-scoped and cannot call `CreateBucket`, which rclone attempts as a preflight on uploads unless this is set):

```
rclone config update r2 no_check_bucket true
```

The resulting block in `~/.config/rclone/rclone.conf` should look like:

```
[r2]
type = s3
provider = Cloudflare
access_key_id = ...
secret_access_key = ...
region = auto
endpoint = https://<account-id>.r2.cloudflarestorage.com
no_check_bucket = true
```

## Point the Go tool at the bucket

Add two lines to the repo-root `.env` (gitignored):

```
JANGLISH_R2_REMOTE=r2:janglish-recordings
JANGLISH_SITE_BASE=https://example.github.io/janglish-record
```

`JANGLISH_SITE_BASE` is a placeholder until the frontend is deployed; `gen-assignment` uses it to construct the share URL.

## Verify

```
rclone lsd r2:janglish-recordings
rclone lsf r2:janglish-recordings
```

The first command should succeed silently (empty bucket); the second should print nothing. If either errors, re-check the endpoint URL and token scope.

From the repo root:

```
just gen-assignment alice wanted-001 wanted-002
just list-progress
```

`list-progress` should show `alice` with `assigned=2 staged=0 done=0`.
