"""Read-only MCP stdio tools. Paths are local to the operator's machine."""
import json
import sys

from .core import Checkpoint, scan

TOOLS = [
    {"name": "inspect_checkpoint", "description": "Read safetensors tensor names, shapes and dtypes without execution",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "assess_checkpoint", "description": "Full numerical health scan with optional trusted checkpoint comparison",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "reference": {"type": "string"}}, "required": ["path"]}}
]


def handle(request):
    method, params = request.get("method"), request.get("params", {})
    if method == "initialize":
        return {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                "serverInfo": {"name": "llmium", "version": "0.1.0"}}
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        args = params.get("arguments", {})
        try:
            if params["name"] == "inspect_checkpoint":
                cp = Checkpoint(args["path"])
                result = {name: {key: tensor[key] for key in ["shape", "dtype", "size", "bytes"]}
                          for name, tensor in cp.tensors.items()}
            elif params["name"] == "assess_checkpoint":
                result = scan(args["path"], args.get("reference"))
            else:
                raise ValueError("Unknown tool")
            return {"content": [{"type": "text", "text": json.dumps(result, allow_nan=False)}]}
        except (ValueError, OSError, KeyError, TypeError) as error:
            return {"isError": True, "content": [{"type": "text", "text": str(error)}]}
    raise ValueError("Unsupported method")


def run():
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            if "id" not in request:
                continue
            response = {"jsonrpc": "2.0", "id": request["id"], "result": handle(request)}
        except (ValueError, TypeError, KeyError) as error:
            response = {"jsonrpc": "2.0", "id": request.get("id") if isinstance(request, dict) else None,
                        "error": {"code": -32600, "message": str(error)}}
        print(json.dumps(response), flush=True)
