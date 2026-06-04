"""add odds_snapshots table

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-05-29
"""
from alembic import op
import sqlalchemy as sa

revision = 'f6a7b8c9d0e1'
down_revision = 'e5f6a7b8c9d0'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'odds_snapshots',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('player_a_id', sa.Integer, sa.ForeignKey('players.id'), nullable=False),
        sa.Column('player_b_id', sa.Integer, sa.ForeignKey('players.id'), nullable=False),
        sa.Column('match_date', sa.Date, nullable=True),
        sa.Column('odds_a', sa.Float, nullable=False),
        sa.Column('odds_b', sa.Float, nullable=False),
        sa.Column('recorded_at', sa.DateTime),
    )
    op.create_index(
        'ix_odds_snap_key',
        'odds_snapshots',
        ['player_a_id', 'player_b_id', 'match_date'],
    )


def downgrade():
    op.drop_index('ix_odds_snap_key', table_name='odds_snapshots')
    op.drop_table('odds_snapshots')
