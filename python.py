import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Khởi tạo Session State cho Lịch sử Chat ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
# Khởi tạo Session State để lưu dữ liệu đã xử lý (dùng làm Context cho Chatbot)
if "df_processed_markdown" not in st.session_state:
    st.session_state["df_processed_markdown"] = None

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý giá trị 0 cho mẫu số
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini (Dùng cho Chức năng Nhận xét tự động) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        # Tái sử dụng logic client đã có
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, khả năng thanh toán, **hiệu quả sử dụng tài sản**, và **khả năng sinh lời**.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# Khởi tạo giá trị mặc định cho chỉ số thanh toán
thanh_toan_hien_hanh_N = "N/A"
thanh_toan_hien_hanh_N_1 = "N/A"

# Khởi tạo giá trị mặc định cho các chỉ số mới (Hiệu quả & Lợi nhuận)
tat_N = "N/A" # Vòng quay Tổng tài sản Năm sau (chỉ tính cho năm N)
npm_N = "N/A" # Biên lợi nhuận ròng Năm sau
npm_N_1 = "N/A" # Biên lợi nhuận ròng Năm trước
roa_N = "N/A" # Tỷ suất sinh lời trên Tổng tài sản Năm sau (chỉ tính cho năm N)


if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- LƯU DỮ LIỆU ĐÃ XỬ LÝ VÀO SESSION STATE CHO KHUNG CHAT SỬ DỤNG ---
            # Dữ liệu được lưu dưới dạng Markdown để dễ dàng đưa vào context của AI
            st.session_state['df_processed_markdown'] = df_processed.to_markdown(index=False, floatfmt=".2f")
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            try:
                # --- I. Chỉ số Thanh toán Hiện hành (Existing) ---
                
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán, tránh chia cho 0
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N != 0 else float('inf')
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else float('inf')

                # --- II. Lấy các chỉ tiêu cần thiết cho Hiệu suất và Lợi nhuận (MỚI) ---
                
                # Tổng Tài sản
                ts_N = df_processed[df_processed['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]['Năm sau'].iloc[0]
                ts_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]['Năm trước'].iloc[0]
                
                # Doanh thu thuần (Giả định có chỉ tiêu này)
                dt_N = df_processed[df_processed['Chỉ tiêu'].str.contains('DOANH THU THUẦN', case=False, na=False)]['Năm sau'].iloc[0]
                dt_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('DOANH THU THUẦN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lợi nhuận sau thuế (Giả định có chỉ tiêu này)
                lnst_N = df_processed[df_processed['Chỉ tiêu'].str.contains('LỢI NHUẬN SAU THUẾ', case=False, na=False)]['Năm sau'].iloc[0]
                lnst_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('LỢI NHUẬN SAU THUẾ', case=False, na=False)]['Năm trước'].iloc[0]
                
                # Tổng Tài sản bình quân (TSBQ) - Chỉ dùng cho năm hiện tại (N)
                tsbq_N = (ts_N + ts_N_1) / 2 if ts_N is not None and ts_N_1 is not None else 0
                
                # --- III. Tính Chỉ số Hiệu quả sử dụng Tài sản (MỚI) ---
                # 1. Vòng quay Tổng tài sản (TAT) = DT thuần / Tổng tài sản bình quân
                if tsbq_N != 0 and dt_N is not None:
                    tat_N = dt_N / tsbq_N
                    
                # --- IV. Tính Chỉ tiêu Lợi nhuận (MỚI) ---
                # 2. Biên lợi nhuận ròng (NPM) = LNST / DT thuần
                if dt_N != 0 and lnst_N is not None:
                    npm_N = (lnst_N / dt_N) * 100
                if dt_N_1 != 0 and lnst_N_1 is not None:
                    npm_N_1 = (lnst_N_1 / dt_N_1) * 100
                    
                # 3. Tỷ suất sinh lời trên Tổng tài sản (ROA) = LNST / Tổng tài sản bình quân
                if tsbq_N != 0 and lnst_N is not None:
                    roa_N = (lnst_N / tsbq_N) * 100

                # --- Hiển thị kết quả (Sử dụng 3 cột mới) ---
                col_thanh_toan, col_hieu_suat, col_loi_nhuan = st.columns(3)

                # Col 1: Khả năng Thanh toán
                with col_thanh_toan:
                    st.markdown("##### 1. Khả năng Thanh toán")
                    st.metric(
                        label="Thanh toán Hiện hành (N-1)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if thanh_toan_hien_hanh_N_1 != float('inf') else "Vô hạn"
                    )
                    st.metric(
                        label="Thanh toán Hiện hành (N)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if thanh_toan_hien_hanh_N != float('inf') else "Vô hạn",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}" if (thanh_toan_hien_hanh_N != float('inf') and thanh_toan_hien_hanh_N_1 != float('inf')) else None
                    )
                    
                # Col 2: Hiệu quả sử dụng Tài sản (MỚI)
                with col_hieu_suat:
                    st.markdown("##### 2. Hiệu quả sử dụng Tài sản")
                    if tat_N != "N/A":
                        st.metric(
                            label="Vòng quay Tổng tài sản (TAT) (N)",
                            value=f"{tat_N:.2f} lần"
                        )
                    else:
                        st.metric(label="Vòng quay Tổng tài sản (TAT) (N)", value="Thiếu DT/TSBQ")
                        
                # Col 3: Chỉ tiêu Lợi nhuận (MỚI)
                with col_loi_nhuan:
                    st.markdown("##### 3. Chỉ tiêu Lợi nhuận")
                    # NPM N
                    if npm_N != "N/A" and npm_N_1 != "N/A":
                        delta_npm = npm_N - npm_N_1
                        st.metric(
                            label="Biên lợi nhuận ròng (NPM) (N)",
                            value=f"{npm_N:.2f}%",
                            delta=f"{delta_npm:.2f}"
                        )
                    else:
                        st.metric(label="Biên lợi nhuận ròng (NPM) (N)", value="Thiếu LNST/DT")
                        
                    # ROA N
                    if roa_N != "N/A":
                        st.metric(
                            label="Tỷ suất sinh lời trên Tổng tài sản (ROA) (N)",
                            value=f"{roa_N:.2f}%"
                        )
                    else:
                        st.metric(label="Tỷ suất sinh lời trên Tổng tài sản (ROA) (N)", value="Thiếu LNST/TSBQ")
                
            except IndexError:
                st.warning("Thiếu một trong các chỉ tiêu: 'TÀI SẢN NGẮN HẠN', 'NỢ NGẮN HẠN', 'DOANH THU THUẦN', 'LỢI NHUẬN SAU THUẾ' hoặc 'TỔNG CỘNG TÀI SẢN' để tính chỉ số.")
                # Đảm bảo các biến vẫn là "N/A"
                pass
            except ZeroDivisionError:
                st.warning("Lỗi chia cho 0 khi tính chỉ số. Vui lòng kiểm tra Nợ ngắn hạn, Doanh thu thuần hoặc Tổng tài sản bình quân.")
                # Đảm bảo các biến vẫn là "N/A"
                pass
            
            # --- Chức năng 5: Nhận xét AI (Tự động) ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI Tự động)")
            
            # Chuẩn bị dữ liệu để gửi cho AI (ĐÃ CẬP NHẬT)
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)',
                    'Vòng quay Tổng tài sản (TAT) (N)',
                    'Biên lợi nhuận ròng (NPM) (N-1)',
                    'Biên lợi nhuận ròng (NPM) (N)',
                    'Tỷ suất sinh lời trên TS (ROA) (N)'
                ],
                'Giá trị': [
                    st.session_state['df_processed_markdown'], # Lấy dữ liệu từ session state
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if not isinstance(thanh_toan_hien_hanh_N, str) else "N/A", 
                    f"{thanh_toan_hien_hanh_N_1:.2f}" if not isinstance(thanh_toan_hien_hanh_N_1, str) else "N/A", 
                    f"{thanh_toan_hien_hanh_N:.2f}" if not isinstance(thanh_toan_hien_hanh_N, str) else "N/A",
                    f"{tat_N:.2f}" if not isinstance(tat_N, str) else "N/A",
                    f"{npm_N_1:.2f}%" if not isinstance(npm_N_1, str) else "N/A",
                    f"{npm_N:.2f}%" if not isinstance(npm_N, str) else "N/A",
                    f"{roa_N:.2f}%" if not isinstance(roa_N, str) else "N/A"
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích"):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")

# =========================================================================
# --- CHỨC NĂNG 6: KHUNG CHAT AI (MỚI THÊM) ---
# =========================================================================

# Hàm khởi tạo Gemini Client
def get_gemini_client():
    """Khởi tạo và trả về Gemini Client."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return None, "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra Streamlit Secrets."
    try:
        client = genai.Client(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"Lỗi khởi tạo Client: {e}"

# Đặt khung chat vào Sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("6. Hỏi đáp với Gemini AI")
    st.write("Đặt câu hỏi cho AI về dữ liệu đã tải lên hoặc bất kỳ kiến thức tài chính nào.")

    # Lấy client và kiểm tra lỗi API Key
    client, error_message = get_gemini_client()

    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Xử lý input từ người dùng
    if prompt := st.chat_input("Hỏi Gemini AI..."):
        
        # 1. Kiểm tra API Client
        if client is None:
            st.error(error_message)
        else:
            # 2. Thêm tin nhắn người dùng vào lịch sử
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 3. Xây dựng nội dung cho Gemini (bao gồm dữ liệu đã xử lý nếu có)
            context_data = st.session_state['df_processed_markdown']

            if context_data:
                # Prompt có chứa dữ liệu tài chính (dữ liệu đã được lưu ở trên)
                context_prompt = f"""
                Bạn là một chuyên gia phân tích tài chính chuyên nghiệp và am hiểu về các báo cáo tài chính Việt Nam. Dữ liệu tài chính đã được tải lên và xử lý ở dạng markdown được cung cấp dưới đây. Hãy trả lời câu hỏi của người dùng (Question: {prompt}) dựa trên Dữ liệu đã xử lý, các chỉ số đã tính toán, và kiến thức chuyên môn của bạn.

                Dữ liệu đã xử lý (Markdown Table):
                {context_data}

                Chỉ trả lời câu hỏi của người dùng.
                """
            else:
                # Prompt chung chung, không có dữ liệu tải lên
                context_prompt = f"Bạn là một trợ lý AI. Hãy trả lời câu hỏi của người dùng. Câu hỏi: {prompt}"

            # 4. Gọi Gemini API
            with st.chat_message("assistant"):
                with st.spinner("Gemini đang trả lời..."):
                    try:
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=[
                                {
                                    "role": "user",
                                    "parts": [{"text": context_prompt}]
                                }
                            ]
                        )
                        ai_response = response.text
                    except APIError as e:
                        ai_response = f"Lỗi gọi Gemini API: {e}"
                    except Exception as e:
                        ai_response = f"Lỗi không xác định: {e}"

                st.markdown(ai_response)
            
            # 5. Thêm tin nhắn AI vào lịch sử
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
