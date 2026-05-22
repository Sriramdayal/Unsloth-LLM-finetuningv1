if __name__ == "__main__":
    import platform

    import uvicorn

    reload_flag = platform.system() != "Windows"
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=reload_flag,
    )
