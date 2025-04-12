import streamlit as st
import pandas as pd
import numpy as np
import os

# =============================
# Đọc dữ liệu sản phẩm
df_products = pd.read_csv(
    'C:\\Users\\LENOVO\\OneDrive\\Pictures\\Bigdata\\GUI_Project\\GUI_Cosine_similarity_model\\Products_ThoiTrangNam_raw.csv'
)
random_products = df_products.head(n=10)
st.session_state.random_products = random_products

# =============================
# Load 1 dòng cosine từ file batch
def load_cosine_batch(index, batch_size=1000):
    batch_file = f"cosine_batches/cosine_batch_{index // batch_size}.npy"
    batch = np.load(batch_file)
    row_index = index % batch_size
    return batch[row_index]

# =============================
# Lấy đề xuất
def get_recommendations(df, ma_san_pham, nums=5):
    matching_indices = df.index[df['product_id'] == ma_san_pham].tolist()
    if not matching_indices:
        st.warning(f"Không tìm thấy sản phẩm với ID: {ma_san_pham}")
        return pd.DataFrame()
    idx = matching_indices[0]

    sim_row = load_cosine_batch(idx)
    sim_scores = list(enumerate(sim_row))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:nums + 1]  # bỏ chính nó

    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices]

# =============================
# Hiển thị đề xuất
def display_recommended_products(recommended_products, cols=5):
    for i in range(0, len(recommended_products), cols):
        col_set = st.columns(cols)
        for j, col in enumerate(col_set):
            if i + j < len(recommended_products):
                product = recommended_products.iloc[i + j]
                with col:
                    st.write(product['product_name'])
                    expander = st.expander("Mô tả")
                    desc = product['description']
                    truncated = ' '.join(desc.split()[:100]) + '...'
                    expander.write(truncated)

# =============================
# Giao diện
st.image(
    'C:\\Users\\LENOVO\\OneDrive\\Pictures\\Bigdata\\GUI_Project\\GUI_Cosine_similarity_model\\hinh.png',
    use_column_width=True
)

product_options = [(row['product_name'], row['product_id']) for _, row in st.session_state.random_products.iterrows()]
selected_product = st.selectbox("Chọn sản phẩm", options=product_options, format_func=lambda x: x[0])
st.session_state.selected_ma_san_pham = selected_product[1]

if st.session_state.selected_ma_san_pham:
    st.write("Mã sản phẩm:", st.session_state.selected_ma_san_pham)
    selected_row = df_products[df_products['product_id'] == st.session_state.selected_ma_san_pham]

    if not selected_row.empty:
        st.write('#### Bạn vừa chọn:')
        st.write('### ', selected_row['product_name'].values[0])
        desc = selected_row['description'].values[0]
        st.write('##### Thông tin:')
        st.write(' '.join(desc.split()[:100]) + '...')
        st.write('##### Các sản phẩm liên quan:')
        recommendations = get_recommendations(df_products, st.session_state.selected_ma_san_pham, nums=3)
        display_recommended_products(recommendations, cols=3)
    else:
        st.error(f"Không tìm thấy sản phẩm với ID: {st.session_state.selected_ma_san_pham}")