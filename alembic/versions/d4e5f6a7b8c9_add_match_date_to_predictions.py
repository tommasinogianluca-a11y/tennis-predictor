"""add match_date to predictions

Revision ID: d4e5f6a7b8c9
Revises: c3f1a2b4d5e6
Create Date: 2026-05-26
"""
from alembic import op
import sqlalchemy as sa

revision = 'd4e5f6a7b8c9'
down_revision = 'c3f1a2b4d5e6'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('predictions', sa.Column('match_date', sa.DateTime(), nullable=True))


def downgrade():
    op.drop_column('predictions', 'match_date')
