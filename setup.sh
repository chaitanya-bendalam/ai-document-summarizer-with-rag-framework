#!/bin/bash

echo "Setting up AI Document Summarizer..."

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Found Python $python_version"

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

if [ ! -f .env ]; then
    echo "Creating .env configuration file..."
    cp config.example .env
    echo "Please edit .env file with your API keys before running the app"
else
    echo ".env file already exists"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source .venv/bin/activate"
echo "2. Edit .env file with your API keys"
echo "3. Run: streamlit run app.py"
echo "4. Open http://localhost:8501 in your browser"
