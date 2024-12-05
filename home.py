import streamlit as st

st.title('Squat Analyzer')

upload_page = st.Page('upload.py', title='Upload Video')
live_page = st.Page('test.py', title='Gunakan Webcam')

pg = st.navigation([upload_page, live_page])

pg.run()
