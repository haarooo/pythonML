import uvicorn
from fastapi import FastAPI

app = FastAPI()

import controller
app.include_router(controller.router)

if __name__ == '__main__':
    uvicorn.run('app:app',port=8000,host='127.0.0.1', reload=True)