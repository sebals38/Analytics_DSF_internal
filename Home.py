import streamlit as st

st.set_page_config(
    page_title="Nielsen analysis dashboard",
    page_icon="📈"   # windows key + .
)

st.title("Nielsen Data Analysis Dashboard")

st.markdown(
    """
            
Here are the categoriy dashboards available:
            
- [ ] [I&R](/I&R_withDB)
- [ ] [CM](/CM_withDB)
- [ ] [Capsule](/Capsule_withDB)
- [ ] [PC](/PC_withDB)
"""
)