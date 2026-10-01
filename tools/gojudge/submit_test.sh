#!/bin/bash
# submit one solution file to the official judge on $PORT; prints the result JSON (no per-case payload)
PORT=$1; PID=$2; SRC=$3
sid=$(curl -s -X POST 127.0.0.1:$PORT/submit -F pid=$PID -F lang=cpp -F code=@$SRC | python3 -c 'import sys,json;print(json.load(sys.stdin)["sid"])')
for i in $(seq 1 200); do r=$(curl -s 127.0.0.1:$PORT/result/$sid); echo "$r" | grep -q '"status":"done"\|"status":"error"' && break; sleep 2; done
echo "$r" | python3 -c 'import sys,json;d=json.load(sys.stdin);print({k:d.get(k) for k in ("status","score","result","error")}, [c.get("status")[:3]+":"+str(c.get("time")) for c in d.get("cases",[])][:12])'
