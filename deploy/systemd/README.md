# Legacy bare-metal units

These units run `spyder-vps` directly from a venv. The **default** deploy path
is Docker Compose on a dedicated CPU VPS (`../iron-spyder.service` +
`docker-compose.yml`). Prefer that.

Keep these only if you intentionally run without Docker. They still target
`/opt/iron-spyder` and `/var/lib/iron-spyder` — never legacy SPY-DER paths —
and must not be installed on a GPU host.
