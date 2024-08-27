#!/usr/bin/env python
# coding: utf-8

import streamlit as st

def main():
    st.title("Multi-Page Streamlit App")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page", ["Linear Regression Assumptions", "Non-Linear Classification Analysis"])

    if page == "Linear Regression Assumptions":
        # Import page1 code
        import page1
        page1.main()
    elif page == "Non-Linear Classification Analysis":
        # Import page2 code
        import page2
        page2.main()

if __name__ == "__main__":
    main()
