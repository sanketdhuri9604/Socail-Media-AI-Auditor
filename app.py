import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openenv_core.env_server import create_app
from models import AuditAction, AuditObservation
from server.environment import SocialMediaAuditorEnvironment
app = create_app(SocialMediaAuditorEnvironment, AuditAction, AuditObservation)
