import os
import json
from typing import Optional
from dotenv import load_dotenv

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

def tavily_search(query: str, max_results: int = 4, search_depth: str = "advanced") -> list:
    load_dotenv()
    """
    Công cụ tìm kiếm sử dụng Tavily, phụ trợ hữu ích để agent tìm kiếm thông tin về các bài báo nghiên cứu (research papers).
    
    Args:
        query (str): Từ khóa hoặc câu truy vấn tìm kiếm (ví dụ: "latest research on LLM agents arxiv").
        max_results (int): Số lượng kết quả tối đa muốn trả về. Mặc định là 4.
        search_depth (str): "basic" hoặc "advanced". "advanced" phù hợp cho research cần thông tin chất lượng cao.
        
    Returns:
        list: Danh sách các URL tìm thấy.
    """
    if TavilyClient is None:
        print("Lỗi: Thiếu thư viện 'tavily-python'. Vui lòng cài đặt bằng lệnh: pip install tavily-python")
        return []
        
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("Lỗi: Không tìm thấy TAVILY_API_KEY trong biến môi trường. Vui lòng cấu hình API key.")
        return []
        
    try:
        client = TavilyClient(api_key=api_key)
        
        # Gọi public API của Tavily
        response = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
        )
        
        results = response.get("results", [])
        return [res.get("url") for res in results if res.get("url")]
        
    except Exception as e:
        print(f"Đã xảy ra lỗi khi gọi Tavily API: {str(e)}")
        return []