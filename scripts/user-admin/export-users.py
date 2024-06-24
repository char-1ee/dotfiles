import argparse
import json


def export_users(uid_start: int, uid_end: int):
    users = {}
    with open("/etc/passwd") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            name, passwd, uid, gid, full_name, home, shell = line.split(":")
            if int(uid) < uid_start or int(uid) >= uid_end:
                continue
            users[name] = {
                "passwd": passwd,
                "uid": uid,
                "gid": gid,
                "full_name": full_name,
                "home": home,
                "shell": shell,
            }
    return users


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid-start", type=int, default=1000)
    parser.add_argument("--uid-end", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(export_users(args.uid_start, args.uid_end)))
