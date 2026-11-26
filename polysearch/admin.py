"""Offline account provisioning and one-time SQLite to PostgreSQL migration."""
import argparse
import getpass
import json
import sqlite3
import sys
from contextlib import closing

from .security import Accounts, ROLES
from .storage import Store, database_location


def migrate(source, destination):
    if not destination.db.postgres:
        raise ValueError("Migration destination must be PostgreSQL")
    with closing(sqlite3.connect(source)) as db:
        db.execute("BEGIN")
        state = db.execute("SELECT id,payload FROM state").fetchall()
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='pilot_campaigns'").fetchone()
        campaigns = db.execute("SELECT id,payload FROM pilot_campaigns").fetchall() if exists else []
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='integration_inbox'").fetchone()
        inbox = db.execute("SELECT id,source_system,external_id,payload FROM integration_inbox").fetchall() if exists else []
    for _, payload in state + campaigns:
        json.loads(payload)
    for _, _, _, payload in inbox:
        json.loads(payload)
    with destination.db.transaction() as db:
        if db.execute("SELECT 1 FROM state LIMIT 1").fetchone() or db.execute("SELECT 1 FROM pilot_campaigns LIMIT 1").fetchone() or db.execute("SELECT 1 FROM integration_inbox LIMIT 1").fetchone():
            raise ValueError("Destination contains data; refusing to overwrite")
        for id, payload in state:
            db.execute("INSERT INTO state VALUES (?, ?)", (id, payload))
        for id, payload in campaigns:
            db.execute("INSERT INTO pilot_campaigns VALUES (?, ?)", (id, payload))
        for row in inbox:
            db.execute("INSERT INTO integration_inbox VALUES (?,?,?,?)", row)
    return {"workspace_rows": len(state), "campaigns": len(campaigns), "inbox_items": len(inbox)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    user = sub.add_parser("user")
    user.add_argument("username")
    user.add_argument("--role", choices=ROLES, required=True)
    user.add_argument("--replace", action="store_true")
    user.add_argument("--password-stdin", action="store_true")
    off = sub.add_parser("disable")
    off.add_argument("username")
    move = sub.add_parser("migrate-sqlite")
    move.add_argument("source")
    sub.add_parser("revoke-sessions")
    args = p.parse_args()
    store = Store(database_location())
    if args.command == "migrate-sqlite":
        print(json.dumps(migrate(args.source, store)))
        return
    accounts = Accounts(store.db)
    if args.command == "user":
        password = sys.stdin.readline().rstrip("\r\n") if args.password_stdin else getpass.getpass("Password: ")
        if not args.password_stdin and password != getpass.getpass("Repeat password: "):
            raise SystemExit("Passwords differ")
        print("Account provisioned:", accounts.provision(args.username, password, args.role, args.replace))
    elif args.command == "disable":
        accounts.disable(args.username)
        print("Account disabled")
    else:
        with store.db.transaction() as db:
            db.execute("DELETE FROM sessions")
        print("All sessions revoked")


if __name__ == "__main__":
    main()
