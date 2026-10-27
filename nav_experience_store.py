import json
import sqlite3
import uuid

from typing import Any


class NavExperienceStore:
    """Stores and retrieves browser navigation experiences using SQLite."""

    MAX_RECORDS_PER_FRAMEWORK = 25

    def __init__(self, db_path: str = "nav_experiences.db") -> None:
        self.db_path = db_path
        self._initialize_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS navigation_experiences (
                    uuid TEXT PRIMARY KEY,
                    framework TEXT NOT NULL,
                    successful INTEGER NOT NULL,
                    result_count INTEGER NOT NULL,
                    experiences TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_framework_results
                ON navigation_experiences (
                    framework,
                    result_count DESC
                )
                """
            )

    def store(
        self,
        framework: str,
        result_count: int,
        experiences: Any,
    ) -> str | None:
        """
        Store an experience if result_count > 0.

        Retains at most 25 records per framework, removing the oldest
        records when the limit is exceeded.

        Returns the stored UUID, or None if the experience is not stored.
        """
        if result_count <= 0:
            return None

        experience_uuid = str(uuid.uuid4())
        experiences_json = json.dumps(experiences, ensure_ascii=False)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO navigation_experiences (
                    uuid,
                    framework,
                    successful,
                    result_count,
                    experiences
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    experience_uuid,
                    framework,
                    1,
                    result_count,
                    experiences_json,
                ),
            )

            # Keep only the 25 most recently stored records
            # for this framework.
            conn.execute(
                """
                DELETE FROM navigation_experiences
                WHERE framework = ?
                  AND uuid NOT IN (
                      SELECT uuid
                      FROM navigation_experiences
                      WHERE framework = ?
                      ORDER BY created_at DESC, rowid DESC
                      LIMIT ?
                  )
                """,
                (
                    framework,
                    framework,
                    self.MAX_RECORDS_PER_FRAMEWORK,
                ),
            )

        return experience_uuid

    def retrieve(
        self,
        framework: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve experiences for a framework, ordered by result_count.

        Args:
            framework: 'Workday', 'Greenhouse', or 'custom-<domain>'.
            limit: Maximum number of experiences to return.

        Returns:
            A list of dictionaries containing the experience metadata
            and decoded JSON experience.
        """
        if limit <= 0:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT uuid, framework, successful, result_count,
                       experiences, created_at
                FROM navigation_experiences
                WHERE framework = ?
                  AND successful = 1
                ORDER BY result_count DESC, created_at DESC
                LIMIT ?
                """,
                (framework, limit),
            ).fetchall()

        return [
            {
                "uuid": row["uuid"],
                "framework": row["framework"],
                "successful": bool(row["successful"]),
                "result_count": row["result_count"],
                "experiences": json.loads(row["experiences"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]