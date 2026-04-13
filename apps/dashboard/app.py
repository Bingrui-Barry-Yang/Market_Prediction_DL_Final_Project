import streamlit as st

from src.config.settings import get_settings


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="Bitcoin Trust Dashboard", layout="wide")
    st.title("Bitcoin Trust-Weighted Trend Dashboard")
    st.write("Environment:", settings.app_env)
    st.info("Dashboard scaffold is ready. Add charts and model monitoring views here.")


if __name__ == "__main__":
    main()
