# Deploying genome to the cloud

Self-hosting recipes for the most common platforms. Pick whichever's easiest; they all give you a HTTPS-reachable genome server in ~5-10 minutes.

genome is a library first, but it ships a REST server (`python -m genome.server`) and a Dockerfile so any container platform works. Below: the four I've verified this month, from lowest-friction to most-configurable.

---

## Fly.io (free tier, ~5 min)

Fly's free tier includes 3 small VMs + 3 GB volumes + 160 GB bandwidth. Good for alpha / internal tools.

```bash
# Install flyctl if you don't have it
curl -L https://fly.io/install.sh | sh

# From the genome repo root:
flyctl launch --no-deploy --name my-genome --region iad
# Accept Dockerfile detection, decline Postgres (SQLite is fine to start)

# Attach a volume for SQLite persistence
flyctl volumes create genome_data --region iad --size 1

# Set env vars
flyctl secrets set \
  GENOME_STORAGE=/data/memory.db \
  GENOME_API_KEY=$(openssl rand -hex 32)

# Edit fly.toml to mount the volume
cat >> fly.toml <<EOF
[[mounts]]
  source = "genome_data"
  destination = "/data"
EOF

flyctl deploy
flyctl open
```

Your server is now at `https://my-genome.fly.dev/docs` with SQLite persistence.

**To upgrade to Postgres** (recommended beyond ~10k memories/user):
```bash
flyctl postgres create --name my-genome-pg
flyctl postgres attach my-genome-pg -a my-genome
# Fly sets DATABASE_URL automatically; rewrite it:
flyctl secrets set GENOME_STORAGE="$(flyctl ssh console -C 'echo $DATABASE_URL')"
```
Note: the pgvector extension needs to be enabled manually on the Fly postgres instance via `flyctl postgres connect` then `CREATE EXTENSION vector;`.

---

## Railway (~3 min, easiest for beginners)

Railway has the lowest barrier — click-deploy with a Docker template detected from the repo.

1. `railway login`
2. `railway init` in the genome repo, link to a new project.
3. `railway add --plugin postgresql` (provisions pg, gets you `DATABASE_URL`).
4. Set variables in the dashboard:
   ```
   GENOME_STORAGE = ${{Postgres.DATABASE_URL}}
   GENOME_API_KEY = <random 32-char hex>
   GENOME_HOST = 0.0.0.0
   GENOME_PORT = 8080
   ```
5. Deploy: `railway up`.
6. `railway domain` — generates a public URL.

Important: Railway's Postgres plugin doesn't have pgvector preinstalled. Run this via `railway connect postgresql`:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Cost: Railway's free tier covers small workloads (~$5/mo credit). Beyond that, pay-as-you-go.

---

## Render (~5 min, clean dashboards)

Render's Blueprint system lets you define infra-as-code.

Create `render.yaml` at the repo root:

```yaml
services:
  - type: web
    name: genome
    runtime: docker
    plan: starter          # $7/mo; free tier doesn't support Docker
    envVars:
      - key: GENOME_STORAGE
        fromDatabase:
          name: genome-pg
          property: connectionString
      - key: GENOME_API_KEY
        generateValue: true
      - key: GENOME_HOST
        value: "0.0.0.0"
      - key: GENOME_PORT
        value: "8080"
    healthCheckPath: /health

databases:
  - name: genome-pg
    plan: starter          # $7/mo; pgvector supported
    postgresMajorVersion: 16
```

Then push the repo + click "New Blueprint" in the Render dashboard. Render builds the Dockerfile and wires everything up.

For pgvector, Render supports it on Postgres 15+ — enable it the first time via:
```sql
CREATE EXTENSION vector;
```
(Render's psql shell is reachable from the database page.)

---

## AWS App Runner (~10 min, production-grade)

For when you want proper auto-scaling, observability, and VPC integration.

```bash
# Publish the image to ECR
aws ecr create-repository --repository-name genome
docker build -t genome:v1.1.0rc1 .
docker tag genome:v1.1.0rc1 $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/genome:v1.1.0rc1
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/genome:v1.1.0rc1

# Create RDS with pgvector (once)
aws rds create-db-instance \
  --db-instance-identifier genome-pg \
  --engine postgres --engine-version 16 \
  --master-username genome --master-user-password $PG_PASS \
  --db-instance-class db.t4g.micro --allocated-storage 20
# Enable pgvector via RDS parameter group: shared_preload_libraries = 'vector'
# Then connect and: CREATE EXTENSION vector;

# Create App Runner service
aws apprunner create-service --service-name genome \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "'$ACCOUNT'.dkr.ecr.'$REGION'.amazonaws.com/genome:v1.1.0rc1",
      "ImageConfiguration": {
        "RuntimeEnvironmentVariables": {
          "GENOME_STORAGE": "postgresql://genome:'$PG_PASS'@'$RDS_HOST':5432/postgres",
          "GENOME_API_KEY": "'$(openssl rand -hex 32)'",
          "GENOME_HOST": "0.0.0.0",
          "GENOME_PORT": "8080"
        },
        "Port": "8080"
      },
      "ImageRepositoryType": "ECR"
    }
  }' \
  --instance-configuration "InstanceRoleArn=$INSTANCE_ROLE,Cpu=1024,Memory=2048"
```

App Runner auto-scales 0→25 instances by default, auto-renews TLS, and integrates with AWS observability. Cost: free tier includes 200K requests + some CPU-hours.

---

## Self-hosted via docker-compose (10 min on any VPS)

The simplest "I have a VPS and want full control" path. Already shipped in the repo as `docker-compose.yml`:

```bash
# On a VPS (anything with Docker + docker-compose)
git clone <internal-url> genome
cd genome

# Edit .env with your API key
cat > .env <<EOF
GENOME_API_KEY=$(openssl rand -hex 32)
EOF

docker-compose up -d
# genome is now at http://<your-vps>:8080, Postgres at :5432
```

Put nginx or Caddy in front for TLS and a domain.

---

## Choosing

| Platform | Best for | Friction | Postgres included | Cost to start |
|---|---|---|---|---|
| **Fly.io** | hobby / alpha / global edge | low | separate addon | $0 (free tier) |
| **Railway** | fastest beginner path | lowest | one click | ~$5/mo credit free |
| **Render** | clean UI, blueprint-as-code | low | one declaration | $7 web + $7 db |
| **AWS App Runner** | production / enterprise | high | RDS separately | pay-per-request |
| **Self-hosted compose** | full control | medium | bundled | VPS cost ($5-20/mo) |

For the vast majority of "I want to try this out today" cases, pick **Railway** or **Fly.io**. For "I'm building a real product", App Runner or a self-hosted VPS with docker-compose is easier to reason about long-term.

---

## Essential env vars (all platforms)

```
GENOME_STORAGE        SQLite path, ":memory:", or postgresql:// DSN
GENOME_API_KEY        Required in X-API-Key header (optional but strongly recommended)
GENOME_EMBED_MODEL    sentence-transformers model (default: all-MiniLM-L6-v2)
GENOME_HOST           Bind address (default: 0.0.0.0 in Docker, 127.0.0.1 locally)
GENOME_PORT           Port (default: 8080)
GENOME_CACHE_SIZE     Response cache LRU capacity (default: 1024)
GENOME_MAX_REQUEST_BYTES   HTTP body limit (default: 1 MiB)
GENOME_LAZY_INIT      Set to "1" to defer model load until first request
```

See [`docs/troubleshooting.md`](troubleshooting.md) if something goes sideways.

## After deployment

Client-side connection (Python):
```python
from genome import Memory
# Or via REST (a new client that hits your HTTPS server):
# Not yet a Python client library, but you can use requests / httpx.
```

Client-side connection (TypeScript):
```ts
import { Memory } from "@frostbyte/genome-memory";
const mem = new Memory({
  baseUrl: "https://my-genome.fly.dev",
  apiKey: process.env.GENOME_API_KEY,
});
```

If the server is reachable and the API key is set, you're done.
