PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS clips (
  id INTEGER PRIMARY KEY,
  engine TEXT NOT NULL,            -- 'piper' | 'coqui' | 'eleven'
  model TEXT NOT NULL,             -- e.g. coqui model name, piper file_id, eleven model_id
  voice TEXT,                      -- e.g. locale/voice/quality or eleven voice name
  lang TEXT,
  text_original TEXT NOT NULL,
  text_normalized TEXT NOT NULL,
  path TEXT NOT NULL UNIQUE,
  sample_rate INTEGER,
  duration_s REAL,
  hash_sha1 TEXT,
  params_json TEXT,
  created_at INTEGER DEFAULT (strftime('%s','now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS clips_fts
USING fts5(text_normalized, content='clips', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS clips_ai AFTER INSERT ON clips
BEGIN
  INSERT INTO clips_fts(rowid, text_normalized) VALUES (new.id, new.text_normalized);
END;

CREATE TRIGGER IF NOT EXISTS clips_ad AFTER DELETE ON clips
BEGIN
  INSERT INTO clips_fts(clips_fts, rowid, text_normalized) VALUES ('delete', old.id, old.text_normalized);
END;

CREATE TRIGGER IF NOT EXISTS clips_au AFTER UPDATE ON clips
BEGIN
  INSERT INTO clips_fts(clips_fts, rowid, text_normalized) VALUES ('delete', old.id, old.text_normalized);
  INSERT INTO clips_fts(rowid, text_normalized) VALUES (new.id, new.text_normalized);
END;
