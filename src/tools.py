#Dict, Any helps us to strictly define nested dictionaries
from typing import Dict, Any

#Simulated production database regional incident
REGIONAL_STATUS_DB: Dict[str, Dict[str, Any]] = {
    "eu-west-1": {
        "status": "degraded",
        "error_code": "502 Bad Gateway",
        "downtime_minutes": 20,
        "incident_id": "INC-4091"
    },
    "us-east-1": {
        "status": "outage",
        "error_code": "503 Service Unavailable",
        "downtime_minutes": 65,
        "incident_id": "INC-4092"
    },
    "ap-south-1": {
        "status": "operational",
        "error_code": None,
        "downtime_minutes": 0,
        "incident_id": None
    }
}

def check_service_status(region: str) -> str:
    """Read-only tool: Checks the real-time outage duration for a cloud region."""
    clean_region = region.lower().strip()
    data = REGIONAL_STATUS_DB.get(clean_region)

    if not data:
        return f"Region '{region}' not found. Valid regions: eu-west-1, us-east-1, ap-south-1."
    return (
        f"Region: {clean_region} | Status: {data['status']} | "
        f"Downtime: {data['downtime_minutes']} minutes | Error: {data['error_code']}"
    )


def issue_sla_credit(account_id: str, amount_usd: float) -> str:
    """Write tool (High Financial Risk): Issues a credit to a customer account."""
    return f"SUCCESS: Issued ${amount_usd:.2f} SLA credit to Account {account_id}."

