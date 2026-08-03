# Support

## Getting help

| What you need | Where to go |
|---|---|
| A bug, crash, or wrong behaviour | [Open an issue](https://github.com/NORTHTEKDevs/genome/issues) with the version, OS, Python version, and a minimal reproduction |
| A question about usage | [Discussions](https://github.com/NORTHTEKDevs/genome/discussions) |
| Something not working after install | [`docs/troubleshooting.md`](./docs/troubleshooting.md) first — it covers the common ones |
| A security vulnerability | **Do not open a public issue.** See [`SECURITY.md`](./SECURITY.md) |
| API details | [`docs/api_reference.md`](./docs/api_reference.md) |
| Reproducing a benchmark claim | [`benchmarks/RESULTS.md`](./benchmarks/RESULTS.md) |

Before filing an install or environment issue, run:

```bash
python -m genome.verify
```

It prints a pass/fail receipt for the local write path and takes a few seconds. Paste the
output into your issue — it answers most of the questions we would otherwise have to ask.

## What community support means here

GENOME is maintained by a small team. Issues and discussions are answered on a best-effort
basis, with no response-time guarantee. Clear reproductions get answered fastest, and pull
requests are welcome — see [`CONTRIBUTING.md`](./CONTRIBUTING.md).

## Commercial support and GENOME Enterprise

The open-source core is free under Apache-2.0 with no usage restrictions, and it will stay
that way.

For organisations that need more than best-effort community support, **GENOME Enterprise**
is a separate commercial product, self-hosted and licensed per deployment. It is built for
regulated and on-premise buyers — lending, KYC/AML, and clinical operations — who have to
answer to an auditor or examiner for what an AI system knew and when.

It adds, on top of the open core:

- a tamper-evident, hash-chained audit record of every read, write, and denied attempt
- point-in-time reconstruction separating when a fact was *true* from when it was *recorded*
- deterministic compliance reports, including adverse-action decision reports
- retention policies, legal hold, and erasure receipts that prove a record was deleted
  without retaining its contents
- role-based access control with separation of duties, and SSO
- a business-hours support commitment with a defined security-patch target

It runs entirely inside your infrastructure. There is no hosted version, deliberately: the
value is that your data never leaves.

To evaluate it, or for commercial support on the open-source core:
**info@northtek.io**

## What we do not offer

- A hosted or SaaS version of GENOME. Self-hosted only, by design.
- Guaranteed response times on community issues. That is what the commercial tier is for.
- Compliance certification. GENOME Enterprise produces evidence and enforces controls that
  support your compliance programme; no software makes an organisation compliant, and we
  will not claim otherwise.
