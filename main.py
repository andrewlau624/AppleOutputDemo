import streamlit as st
import json
import uuid
from datetime import datetime
from PIL import Image
from worker import QAlignWorker

# Standard Streamlit Config
st.set_page_config(page_title="Vision Systems Gateway", layout="wide")


@st.cache_resource
def get_worker():
    return QAlignWorker()


# --- Header ---
st.title("Data Validation Gateway")
st.caption("Agentic Data Validation System | Prototype v1.2")

# --- Sidebar ---
with st.sidebar:
    st.header("Validation Contract")
    threshold = st.slider("Quality Threshold (0-1)", 0.0, 1.0, 0.45, 0.01)
    op = st.selectbox("Operator", [">=", ">", "<=", "<", "=="])

# --- Ingest ---
uploaded_file = st.file_uploader("Select Image Payload", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    current_rule = {"threshold": threshold, "operator": op}

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Payload")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Decision Engine")

        with st.spinner("Analyzing..."):
            worker = get_worker()
            # This follows your Prisma schema
            db_record = worker.get_signal(img, current_rule)

            # Fill in the ID and timestamps for the demo push
            db_record["id"] = str(uuid.uuid4())
            db_record["createdAt"] = datetime.now().isoformat()
            db_record["updatedAt"] = datetime.now().isoformat()

        # Native Streamlit Status Indicators
        if db_record["status"] == "APPROVED":
            st.success(f"STATUS: {db_record['status']}")
            st.metric("Quality Score", f"{db_record['rawOutputs']['quality_index']:.4f}")
        else:
            st.error(f"STATUS: {db_record['status']}")
            st.metric("Quality Score", f"{db_record['rawOutputs']['quality_index']:.4f}")

        st.divider()

        # --- Database Logic Visualization ---
        st.subheader("Database Transaction")
        with st.status("Pushing to Prisma...", expanded=True) as status:
            st.write(f"Validated: {db_record['isValidated']}")
            st.write(f"Status: {db_record['status']}")
            st.code(f"UPSERT ImageValidation WHERE id = {db_record['id'][:8]}", language="sql")
            status.update(label="Transaction Complete", state="complete")

        # --- Manifest View ---
        with st.expander("View Prisma JSON Object"):
            st.json(db_record)

        st.download_button(
            label="Download Manifest",
            data=json.dumps(db_record, indent=4),
            file_name="validation_output.json",
            mime="application/json"
        )
else:
    st.info("Please upload an image to begin validation.")