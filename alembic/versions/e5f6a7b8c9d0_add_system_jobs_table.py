"""add system_jobs table

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-05-29
"""
from alembic import op
import sqlalchemy as sa

revision = 'e5f6a7b8c9d0'
down_revision = 'd4e5f6a7b8c9'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'system_jobs',
        sa.Column('id', sa.String(8), primary_key=True),
        sa.Column('action', sa.String(50)),
        sa.Column('status', sa.String(20), server_default='running'),
        sa.Column('log', sa.Text, server_default=''),
        sa.Column('started_at', sa.DateTime),
        sa.Column('finished_at', sa.DateTime, nullable=True),
    )


def downgrade():
    op.drop_table('system_jobs')
