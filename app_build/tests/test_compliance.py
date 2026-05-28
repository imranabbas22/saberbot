import pytest
import os
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from main import app

client = TestClient(app)

def test_compliance_endpoint_txt(tmp_path):
    test_file = tmp_path / "test_doc.txt"
    test_file.write_text("The employee must work 60 hours a week and resignation requires 30 days notice.", encoding="utf-8")
    
    with TestClient(app) as client:
        with open(test_file, "rb") as f:
            res = client.post("/api/compliance", files={"file": ("test_doc.txt", f, "text/plain")})
            
        assert res.status_code == 200
        data = res.json()
        assert "overall_score" in data
        assert "findings" in data
        assert len(data["findings"]) > 0
        assert "status" in data["findings"][0]
        assert "reason" in data["findings"][0]
        assert "accountability" in data["findings"][0]

def test_compliance_endpoint_empty(tmp_path):
    test_file = tmp_path / "empty.txt"
    test_file.write_text("", encoding="utf-8")
    
    with TestClient(app) as client:
        with open(test_file, "rb") as f:
            res = client.post("/api/compliance", files={"file": ("empty.txt", f, "text/plain")})
            
        assert res.status_code == 400

from unittest.mock import patch, MagicMock
from routers.compliance import extract_text

def test_extract_text_pdf():
    with patch("pdfplumber.open") as mock_pdf:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "pdf text"
        mock_pdf.return_value.__enter__.return_value.pages = [mock_page]
        assert "pdf text" in extract_text("dummy.pdf", "dummy.pdf")

def test_extract_text_docx():
    with patch("docx.Document") as mock_doc:
        mock_para = MagicMock()
        mock_para.text = "docx text"
        mock_doc.return_value.paragraphs = [mock_para]
        assert "docx text" in extract_text("dummy.docx", "dummy.docx")

def test_extract_text_xlsx():
    with patch("pandas.read_excel") as mock_pd:
        mock_pd.return_value.to_string.return_value = "xlsx text"
        assert "xlsx text" in extract_text("dummy.xlsx", "dummy.xlsx")

def test_extract_text_image():
    with patch("pytesseract.image_to_string") as mock_ocr, patch("PIL.Image.open") as mock_img:
        mock_ocr.return_value = "ocr text"
        assert "ocr text" in extract_text("dummy.jpg", "dummy.jpg")

def test_extract_text_unsupported():
    assert "Unsupported file format" in extract_text("dummy.xyz", "dummy.xyz")
