@echo off
REM Start the backend and frontend servers
start cmd /k "cd backend && python app.py"
start cmd /k "cd frontend && npm run dev"
