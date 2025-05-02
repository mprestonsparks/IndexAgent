from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

# Assumed list of app repos; replace with actual repo names as needed
APP_REPOS = ["market-analysis", "trade-manager", "trade-discovery", "trade-dashboard", "git-books"]

# Directory where baseline audit reports are stored
REPORTS_DIR = "logs/baseline_issues"
SUMMARY_FILE = os.path.join(REPORTS_DIR, "baseline_audit_summary.md")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def collate_reports(**kwargs):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    summary_lines = ["# Baseline Audit Summary\n"]
    for repo in APP_REPOS:
        report_path = os.path.join(REPORTS_DIR, f"{repo}_baseline_report.md")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                summary_lines.append(f"## {repo}\n")
                summary_lines.extend(f.readlines())
                summary_lines.append("\n")
        else:
            summary_lines.append(f"## {repo}\nNo report found.\n\n")
    with open(SUMMARY_FILE, "w") as f:
        f.writelines(summary_lines)

with DAG(
    dag_id="baseline_audit",
    default_args=default_args,
    description="Phase 0: Baseline Audit for all app repos",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["baseline", "audit"],
) as dag:

    audit_tasks = []
    for repo in APP_REPOS:
        task = BashOperator(
            task_id=f"audit_{repo}",
            bash_command=f"docker exec indexagent ./scripts/baseline_audit.sh /repos/{repo}",
            env={"REPO_NAME": repo},
        )
        audit_tasks.append(task)

    collate = PythonOperator(
        task_id="collate_reports",
        python_callable=collate_reports,
        provide_context=True,
    )

    audit_tasks >> collate