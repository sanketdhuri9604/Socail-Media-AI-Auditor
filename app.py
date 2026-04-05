import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_app
from models import AuditAction, AuditObservation
from server.environment import SocialMediaAuditorEnvironment

env = SocialMediaAuditorEnvironment()
app = create_app(env, AuditAction, AuditObservation)