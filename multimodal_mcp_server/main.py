from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")

@mcp.tool
def flip_image(artifact_filename: str) -> dict:
    print(artifact_filename)
    return {"status": "success", "message": "Image successfully flipped", "original_filename": artifact_filename}

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)