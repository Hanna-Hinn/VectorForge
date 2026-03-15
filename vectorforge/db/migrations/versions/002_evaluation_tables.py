"""Add evaluation tables — evaluation_runs, evaluation_results, recommendations.

Revision ID: 002
Revises: 001
Create Date: 2026-03-15 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str = "001"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    _uuid_col = postgresql.UUID(as_uuid=True)
    _uuid_default = sa.text("gen_random_uuid()")
    _now_default = sa.func.now()

    # --- evaluation_runs ---
    op.create_table(
        "evaluation_runs",
        sa.Column(
            "id",
            _uuid_col,
            server_default=_uuid_default,
            primary_key=True,
        ),
        sa.Column(
            "status",
            sa.String(50),
            server_default="pending",
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "sample_size",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column("config", postgresql.JSONB(), nullable=True),
        sa.Column("summary_scores", postgresql.JSONB(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=_now_default,
            nullable=False,
        ),
    )
    op.create_index(
        "ix_evaluation_runs_status", "evaluation_runs", ["status"]
    )
    op.create_index(
        "ix_evaluation_runs_created_at", "evaluation_runs", ["created_at"]
    )

    # --- evaluation_results ---
    op.create_table(
        "evaluation_results",
        sa.Column(
            "id",
            _uuid_col,
            server_default=_uuid_default,
            primary_key=True,
        ),
        sa.Column(
            "run_id",
            _uuid_col,
            sa.ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "query_log_id",
            _uuid_col,
            sa.ForeignKey("query_logs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("evaluator_name", sa.String(255), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=_now_default,
            nullable=False,
        ),
    )
    op.create_index(
        "ix_evaluation_results_run_id", "evaluation_results", ["run_id"]
    )
    op.create_index(
        "ix_evaluation_results_query_log_id",
        "evaluation_results",
        ["query_log_id"],
    )
    op.create_index(
        "ix_evaluation_results_evaluator_name",
        "evaluation_results",
        ["evaluator_name"],
    )

    # --- recommendations ---
    op.create_table(
        "recommendations",
        sa.Column(
            "id",
            _uuid_col,
            server_default=_uuid_default,
            primary_key=True,
        ),
        sa.Column(
            "run_id",
            _uuid_col,
            sa.ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(50), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("evidence", postgresql.JSONB(), nullable=True),
        sa.Column(
            "status",
            sa.String(50),
            server_default="pending",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=_now_default,
            nullable=False,
        ),
    )
    op.create_index(
        "ix_recommendations_run_id", "recommendations", ["run_id"]
    )
    op.create_index(
        "ix_recommendations_severity", "recommendations", ["severity"]
    )
    op.create_index(
        "ix_recommendations_status", "recommendations", ["status"]
    )


def downgrade() -> None:
    op.drop_table("recommendations")
    op.drop_table("evaluation_results")
    op.drop_table("evaluation_runs")
