import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from streamlit_option_menu import option_menu
import streamlit as st
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder

# Load environment variables (if any)
load_dotenv()

# Load models
priority_model = joblib.load('priority_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Load or initialize tasks
try:
    df = pd.read_csv('tasks.csv', parse_dates=['Deadline'])
except FileNotFoundError:
    df = pd.DataFrame(columns=[
        'Task ID', 'Task Name', 'Assignee', 'Tag',
        'Priority', 'Status', 'Time Taken (hrs)', 'Deadline',
        'Overdue', 'Progress', 'Comments', 'File Path', 'Description',
        'Processed_Description', 'Days Until Deadline', 'PriorityEncoded'
    ])
    df.to_csv('tasks.csv', index=False)

if 'priorityEncoded' in df.columns and 'PriorityEncoded' not in df.columns:
    df['PriorityEncoded'] = df['priorityEncoded']

# Compute dynamic columns
today = pd.Timestamp.now().normalize()  # normalize to remove time part for date comparison
df['Deadline'] = pd.to_datetime(df['Deadline'], errors='coerce')
df['Overdue'] = df['Deadline'] < today
df['Days Until Deadline'] = (df['Deadline'] - today).dt.days.fillna(0).astype(int)
progress_map = {'Completed': 100, 'In Progress': 50, 'To Do': 0}
df['Progress'] = df['Status'].map(progress_map).fillna(0)

# Encode current priority
priority_encoder = LabelEncoder()
df['PriorityEncoded'] = priority_encoder.fit_transform(df['Priority'].fillna("Low"))

# Streamlit config
st.set_page_config(page_title="Task Manager", layout="wide")
st.markdown("""
    <style>
        body { background-color: #f8f9fc; font-family: 'Segoe UI', sans-serif; }
        .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .card {
            background: white; border-radius: 1rem; padding: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    selected = option_menu("Task Manager", ["Dashboard", "Add Task", "Update Task", "Delete Task", "Predict Priority", "Analytics"],
                           icons=['bar-chart', 'plus-circle', 'pencil', 'trash', 'activity', 'graph-up'],
                           menu_icon="clipboard", default_index=0)

# Ensure session states
if "selected_task" not in st.session_state:
    st.session_state["selected_task"] = None
if "show_more" not in st.session_state:
    st.session_state["show_more"] = {}

def render_metric_card(title, value, subtitle="", icon="📌", color_light="#ffffff", color_dark="#1e1e1e"):
    base_theme = st.get_option("theme.base")
    dark_mode = base_theme == "dark"

    background = color_dark if dark_mode else color_light
    title_color = "#e0e0e0" if dark_mode else "#222222"
    subtitle_color = "#bbbbbb" if dark_mode else "#666666"
    value_color = "#ffffff" if dark_mode else "#000000"

    return f"""
    <div style="
        background-color: {background};
        padding: 0.6rem;
        border-radius: 0.5rem;
        box-shadow: 0 3px 8px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s ease;
        max-width: 180px;
        margin: 0 auto;
    " onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
        <div style="font-size: 1.1rem;">{icon}</div>
        <div style="font-weight: 600; font-size: 0.95rem; color: {title_color};">
            {title}
        </div>
        <div style="font-size: 1.2rem; font-weight: bold; margin: 0.4rem 0; color: {value_color};">
            {value}
        </div>
        <div style="font-size: 0.75rem; color: {subtitle_color};">
            {subtitle}
        </div>
    </div>
    """
def color_priority(val):
    if val == 'High': return 'background-color: #ff9999; color: black;'
    elif val == 'Medium': return 'background-color: #ffcc80; color: black;'
    elif val == 'Low': return 'background-color: #b3ffb3; color: black;'
    return ''

def mini_task_card(row, color):
    base_theme = st.get_option("theme.base")
    dark_mode = base_theme == "dark"
    
    tooltip = f"{row['Task Name']} • Due: {row['Deadline'].date() if pd.notnull(row['Deadline']) else 'N/A'} • Assigned to {row['Assignee']}"
    glow = "0 0 10px rgba(255, 76, 76, 0.4)" if row['Priority'] == "High" else "0 2px 6px rgba(0,0,0,0.08)"
    text_color = "#f2f2f2" if dark_mode else "#222222"
    hover_shadow = "0 6px 12px rgba(255,255,255,0.15)" if dark_mode else "0 6px 12px rgba(0,0,0,0.15)"
    
    return f"""
    <div title="{tooltip}" style="
        background-color: {color};
        color: {text_color};
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 0.75rem;
        animation: fadeIn 0.4s ease-in-out;
        box-shadow: {glow};
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    " onmouseover="this.style.transform='scale(1.02)'; this.style.boxShadow='{hover_shadow}';"
      onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='{glow}';">
        <div style="font-weight:600; font-size: 0.95rem;">🆔 {row['Task ID']} - {row['Task Name']}</div>
        <div style="font-size: 0.85rem; margin-top: 0.25rem;">
            📅 <b>Deadline:</b> {row['Deadline'].date() if pd.notnull(row['Deadline']) else 'N/A'}<br>
            📌 <b>Status:</b> {row['Status']}<br>
            🔥 <b>Priority:</b> {row['Priority']}
        </div>
    </div>
    """

if selected == "Dashboard":
    st.title("📋 Task Manager Dashboard")

    with st.expander("🔍 Filter Tasks", expanded=True):
        status_filter = st.multiselect("Status", df['Status'].dropna().unique(), default=df['Status'].dropna().unique())
        priority_filter = st.multiselect("Priority", df['Priority'].dropna().unique(), default=df['Priority'].dropna().unique())
        assignee_filter = st.multiselect("Assignee", df['Assignee'].dropna().unique(), default=df['Assignee'].dropna().unique())

    filtered_df = df[
        df['Status'].isin(status_filter) &
        df['Priority'].isin(priority_filter) &
        df['Assignee'].isin(assignee_filter)
    ]

        
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            render_metric_card(
                "Total Tasks", len(df), "All tasks in system", "📋", "#ffffff"
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            render_metric_card(
                "Filtered Tasks", len(filtered_df), "After filters applied", "🔍", "#f9f9ff"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            render_metric_card(
                "Overdue Tasks", int(df['Overdue'].sum()), "Deadline passed", "⚠️", "#fff5f5"
            ),
            unsafe_allow_html=True
        )
    
    with col4:
        avg_prog = f"{filtered_df['Progress'].mean():.1f}%" if not filtered_df.empty else "0%"
        st.markdown(
            render_metric_card(
                "Avg. Progress", avg_prog, "Based on filtered", "📈", "#f0faff"
            ),
            unsafe_allow_html=True
        )
    
    # Add a small vertical spacer (gap)
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # Apply color styling to Priority column
    styled_df = filtered_df.style.applymap(color_priority, subset=['Priority'])
    
    # Task list section
    with st.expander("🗂️ Tasks List", expanded=True):
        st.dataframe(styled_df, height=400)

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("📉 Overdue Task Trend")
    
        # Ensure Deadline is datetime
        df['Deadline'] = pd.to_datetime(df['Deadline'], errors='coerce')
    
        # Convert date only (strip time)
        df['Date'] = df['Deadline'].dt.date
    
        # Filter only overdue tasks
        overdue_df = df[df['Overdue'] == True]
    
        # Count overdue tasks by date
        overdue_trend = overdue_df.groupby('Date').size()
    
        # Plot
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(overdue_trend.index, overdue_trend.values, color='#B26E63', linewidth=2, marker='o')
        ax.fill_between(overdue_trend.index, overdue_trend.values, color='#F2D8D0', alpha=0.3)
    
        ax.set_title("Overdue Tasks Over Time", fontsize=13)
        ax.set_ylabel("Number of Overdue Tasks")
        ax.set_xlabel("Deadline Date")
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xticks(overdue_trend.index[::max(1, len(overdue_trend)//10)])  # limit ticks
        st.pyplot(fig)

    with col6:
        st.subheader("📊 Status Breakdown")
    
        status_counts = filtered_df['Status'].value_counts()
        labels = status_counts.index
        sizes = status_counts.values
    
        # 🎨 Pretty pastel colors — adjustable if needed
        pastel_colors = ['#f2b6cd', '#ffe2b7', '#d6edc7']  # To Do, In Progress, Completed (edit order if needed)
    
        fig, ax = plt.subplots(figsize=(6, 5))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=pastel_colors,
            wedgeprops=dict(width=0.55),
            textprops=dict(color="black")
        )
    
        # 🌟 Style the inside percentages
        for autotext in autotexts:
            autotext.set_color("black")
            autotext.set_fontsize(11)
            autotext.set_weight("bold")
    
        # ✍️ Center text: Total Tasks
        total_tasks = int(sum(sizes))
        ax.text(0, 0, f"{total_tasks}\nTasks", ha='center', va='center', fontsize=14, weight='bold', color='#3c2a21')
    
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

    st.subheader("🚨 Smart Suggestions")
    smart_sections = [
        {
            "title": "📆 Due Today / Tomorrow",
            "df": df[(df['Days Until Deadline'] <= 1) & (df['Status'] != 'Completed')],
            "color": "#2e2e2e" if st.get_option("theme.base") == "dark" else "#fff8e1",
            "key": "due"
        },
        {
            "title": "⚠️ Overdue Tasks",
            "df": df[df['Overdue'] & (df['Status'] != 'Completed')],
            "color": "#5c1a1a" if st.get_option("theme.base") == "dark" else "#fdecea",
            "key": "overdue"
        },
        {
            "title": "🔥 High Priority Pending",
            "df": df[(df['Priority'] == 'High') & (df['Status'] == 'To Do')],
            "color": "#402020" if st.get_option("theme.base") == "dark" else "#fff0f0",
            "key": "high"
        }
    ]

    st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

    for section in smart_sections:
        with st.expander(f"{section['title']} ({len(section['df'])})", expanded=False):
            show_all = st.session_state["show_more"].get(section["key"], False)
            data = section["df"]
            display_df = data if show_all else data.head(4)
            if display_df.empty:
                st.info("✅ No tasks in this category.")
            else:
                cols = st.columns(2)
                for idx, (_, row) in enumerate(display_df.iterrows()):
                    col = cols[idx % 2]
                    with col:
                        st.markdown(mini_task_card(row, section["color"]), unsafe_allow_html=True)

            if len(data) > 4:
                btn_label = "Show Less" if show_all else "Show More"
                if st.button(btn_label, key=f"toggle_{section['key']}"):
                    st.session_state["show_more"][section["key"]] = not show_all
                    st.rerun()
                    
    st.subheader("🗓️ Calendar View")
    selected_date = st.date_input("Select a Date", value=datetime.today())
    tasks_on_date = df[df['Deadline'].dt.date == selected_date]
    st.dataframe(tasks_on_date[['Task ID', 'Task Name', 'Assignee', 'Tag', 'Priority', 'Status', 'Deadline']])


elif selected == "Add Task":
    st.title("➕ Add New Task")
    with st.form("add_task_form"):
        new_id = st.text_input("Task ID")
        new_name = st.text_input("Task Name")
        new_assignee_custom = st.text_input("Enter New Assignee (optional)")
        new_assignee_select = st.selectbox("Or Select Existing Assignee", options=[""] + list(df['Assignee'].dropna().unique()))
        new_tag = st.text_input("Tag (optional)")
        new_priority = st.selectbox("Priority", ['High', 'Medium', 'Low'])
        new_status = st.selectbox("Status", ['To Do', 'In Progress', 'Completed'])
        new_time = st.number_input("Time Taken (hrs)", min_value=1, max_value=100)
        new_deadline = st.date_input("Deadline")
        new_desc = st.text_area("Task Description")
        comments = st.text_area("Comments (optional)")
        file_upload = st.file_uploader("Attach a file (optional)", type=['pdf', 'docx', 'txt'])
        add_submit = st.form_submit_button("Add Task")

        if add_submit:
            if not new_id.strip():
                st.error("❌ Task ID cannot be empty")
            elif not new_name.strip():
                st.error("❌ Task Name cannot be empty")
            elif not new_desc.strip():
                st.error("❌ Description cannot be empty")
            elif new_id in df['Task ID'].values:
                st.error("❌ Task ID already exists.")
            else:
                assignee = new_assignee_custom if new_assignee_custom else new_assignee_select
                deadline_ts = pd.to_datetime(new_deadline)
                new_task = pd.DataFrame([{
                    'Task ID': new_id,
                    'Task Name': new_name,
                    'Assignee': assignee,
                    'Tag': new_tag,
                    'Priority': new_priority,
                    'Status': new_status,
                    'Time Taken (hrs)': new_time,
                    'Deadline': deadline_ts,
                    'Comments': comments,
                    'File Path': file_upload.name if file_upload else None,
                    'Description': new_desc,
                    'Processed_Description': new_desc.lower().strip(),
                    'Days Until Deadline': (deadline_ts - today).days,
                    'PriorityEncoded': priority_encoder.transform([new_priority])[0],
                    'Overdue': deadline_ts < today,
                    'Progress': progress_map.get(new_status, 0)
                }])
                df = pd.concat([df, new_task], ignore_index=True)
                df.to_csv("tasks.csv", index=False)
                st.success("✅ Task added successfully!")

elif selected == "Update Task":
    st.title("✏️ Update Task")
    task_id = st.text_input("Enter Task ID to update")
    if task_id:
        row = df[df['Task ID'] == task_id]
        if not row.empty:
            task = row.iloc[0]
            with st.form("update_form"):
                updated_name = st.text_input("Task Name", value=task['Task Name'])
                updated_assignee = st.text_input("Assignee", value=task['Assignee'])
                updated_tag = st.text_input("Tag", value=task['Tag'])
                updated_priority = st.selectbox("Priority", ['High', 'Medium', 'Low'], index=['High', 'Medium', 'Low'].index(task['Priority']))
                updated_status = st.selectbox("Status", ['To Do', 'In Progress', 'Completed'], index=['To Do', 'In Progress', 'Completed'].index(task['Status']))
                updated_time = st.number_input("Time Taken (hrs)", value=int(task['Time Taken (hrs)']))
                updated_deadline = st.date_input("Deadline", value=task['Deadline'].date() if pd.notnull(task['Deadline']) else datetime.today())
                updated_file_upload = st.file_uploader("Attach a file (optional)", type=['pdf', 'docx', 'txt'])
                updated_desc = st.text_area("Description", value=task.get("Description", ""))
                updated_comments = st.text_area("Comments", value=task.get("Comments", ""))
                update_submit = st.form_submit_button("Update Task")

                if update_submit:
                    # Set file path to existing value if no new upload
                    file_path_value = updated_file_upload.name if updated_file_upload else task.get('File Path', None)
                    df.loc[df['Task ID'] == task_id, [
                        'Task Name', 'Assignee', 'Tag', 'Priority', 'Status',
                        'Time Taken (hrs)', 'Deadline', 'Comments', 'Description',
                        'Processed_Description', 'Days Until Deadline', 'File Path', 'PriorityEncoded'
                    ]] = [
                        updated_name, updated_assignee, updated_tag, updated_priority, updated_status,
                        updated_time, pd.to_datetime(updated_deadline), updated_comments, updated_desc,
                        updated_desc.lower().strip(), (pd.to_datetime(updated_deadline) - today).days, file_path_value,
                        priority_encoder.transform([updated_priority])[0]
                    ]
                    df['Overdue'] = df['Deadline'] < today
                    df['Progress'] = df['Status'].map(progress_map).fillna(0)
                    df.to_csv("tasks.csv", index=False)
                    st.success(f"✅ Task '{task_id}' updated successfully!")

elif selected == "Delete Task":
    st.title("🗑️ Delete Task")
    with st.form("delete_task_form"):
        del_id = st.selectbox("Select Task ID to delete", options=df['Task ID'].dropna().unique())
        del_submit = st.form_submit_button("Delete Task")

        if del_submit:
            df = df[df['Task ID'] != del_id]
            df.to_csv("tasks.csv", index=False)
            st.success(f"Task with ID '{del_id}' deleted.")

elif selected == "Predict Priority":
    st.title("🔮 Predict Task Priority by Task ID")
    with st.form("priority_form"):
        task_id = st.text_input("Enter Task ID")
        save_prediction = st.checkbox("📥 Save prediction to dataset?")
        submitted = st.form_submit_button("Predict")
        if submitted:
            row = df[df['Task ID'] == task_id]
            if row.empty:
                st.error("❌ Task not found.")
            else:
                status = row['Status'].values[0]
                time_taken = row['Time Taken (hrs)'].values[0]
                days_left = row['Days Until Deadline'].values[0]
                input_df = pd.DataFrame({
                    'Time Taken (hrs)': [time_taken],
                    'Status': [status],
                    'Days Until Deadline': [days_left]
                })
                X_input = preprocessor.transform(input_df)
                pred = priority_model.predict(X_input)[0]
                st.success(f"🎯 Predicted Priority: **{pred}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("📊 Model Confidence")
                    prediction_probs = priority_model.predict_proba(X_input)[0]
                    priority_labels = priority_model.classes_
                    fig1, ax1 = plt.subplots(figsize=(3, 2))
                    sns.barplot(x=priority_labels, y=prediction_probs, palette=['#ff4c4c', '#ffa500', '#4caf50'], ax=ax1)
                    ax1.set_ylabel("Probability", fontsize=8)
                    ax1.set_title("Confidence", fontsize=9)
                    ax1.tick_params(labelsize=7)
                    st.pyplot(fig1)
                    plt.clf()
                with col2:
                    st.caption("📈 Priority Distribution (w/ prediction)")
                    df_with_pred = pd.concat([df[['Task ID', 'Priority']], pd.DataFrame([{'Task ID': task_id, 'Priority': pred}])], ignore_index=True)
                    priority_counts = df_with_pred['Priority'].value_counts().reindex(['High', 'Medium', 'Low']).fillna(0)
                    fig2, ax2 = plt.subplots(figsize=(3, 2))
                    sns.barplot(x=priority_counts.index, y=priority_counts.values, palette=['#ff4c4c', '#ffa500', '#4caf50'], ax=ax2)
                    ax2.set_ylabel("Count", fontsize=8)
                    ax2.set_title("Distribution", fontsize=9)
                    ax2.tick_params(labelsize=7)
                    st.pyplot(fig2)
                    plt.clf()

                if save_prediction:
                    df.loc[df['Task ID'] == task_id, ['Priority', 'PriorityEncoded']] = [
                        pred, priority_encoder.transform([pred])[0]
                    ]
                    df.to_csv("tasks.csv", index=False)
                    st.success("✅ Prediction saved to dataset.")

elif selected == "Analytics":
    st.title("📊 Task Analytics")
    st.metric("Completion Rate", f"{(df['Status'] == 'Completed').mean() * 100:.1f}%")
    st.metric("Average Time Taken", f"{df['Time Taken (hrs)'].mean():.1f} hrs")
    team_perf = df.groupby('Assignee')['Status'].value_counts().unstack().fillna(0)
    st.bar_chart(team_perf)

# --- Footer ---
# Footer
st.markdown("---")
st.markdown("Developed by **Shruti S. Nayak** | Powered by Streamlit & scikit-learn")