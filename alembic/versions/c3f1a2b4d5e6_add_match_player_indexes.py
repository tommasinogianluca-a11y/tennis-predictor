"""add match player indexes

Revision ID: c3f1a2b4d5e6
Revises: 8437157cd1da
Create Date: 2026-05-23

"""
from alembic import op

revision = 'c3f1a2b4d5e6'
down_revision = '8437157cd1da'
branch_labels = None
depends_on = None


def upgrade():
    op.create_index('ix_match_player1', 'matches', ['player1_id'], unique=False)
    op.create_index('ix_match_player2', 'matches', ['player2_id'], unique=False)
    op.create_index('ix_match_winner', 'matches', ['winner_id'], unique=False)
    op.create_index('ix_match_surface', 'matches', ['surface'], unique=False)
    op.create_index('ix_match_surface_date', 'matches', ['surface', 'date'], unique=False)
    op.create_index('ix_elo_player_surface', 'elo_ratings', ['player_id', 'surface'], unique=False)


def downgrade():
    op.drop_index('ix_match_player1', table_name='matches')
    op.drop_index('ix_match_player2', table_name='matches')
    op.drop_index('ix_match_winner', table_name='matches')
    op.drop_index('ix_match_surface', table_name='matches')
    op.drop_index('ix_match_surface_date', table_name='matches')
    op.drop_index('ix_elo_player_surface', table_name='elo_ratings')
