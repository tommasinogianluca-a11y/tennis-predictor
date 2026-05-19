"""widen score_string, unique atp_id

Revision ID: 8437157cd1da
Revises: 73ba09c2bcb1
Create Date: 2026-05-19 18:33:20.077907

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8437157cd1da'
down_revision: Union[str, None] = '73ba09c2bcb1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Use batch mode for SQLite compatibility (ALTER COLUMN / ADD CONSTRAINT)
    with op.batch_alter_table('matches') as batch_op:
        batch_op.alter_column('score_string',
                              existing_type=sa.VARCHAR(length=100),
                              type_=sa.String(length=200),
                              existing_nullable=True)

    with op.batch_alter_table('players') as batch_op:
        batch_op.create_unique_constraint('uq_players_atp_id', ['atp_id'])


def downgrade() -> None:
    with op.batch_alter_table('players') as batch_op:
        batch_op.drop_constraint('uq_players_atp_id', type_='unique')

    with op.batch_alter_table('matches') as batch_op:
        batch_op.alter_column('score_string',
                              existing_type=sa.String(length=200),
                              type_=sa.VARCHAR(length=100),
                              existing_nullable=True)
