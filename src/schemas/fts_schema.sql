-- This script must be run after other tables are created

-- Using simple tokenizer for Chinese language support
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    id UNINDEXED,
    document_id UNINDEXED,
    chunk_text,
    file,
    page UNINDEXED,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61'
);


-- Add the three auto-sync triggers
-- 1) After INSERT on chunks => Insert a corresponding row in chunks_fts
CREATE TRIGGER IF NOT EXISTS chunks_ai
AFTER INSERT ON chunks
BEGIN
    INSERT INTO chunks_fts(
        rowid,
        document_id,
        chunk_text,
        file,
        page
    )
    VALUES(
        new.id,
        new.document_id,
        new.chunk_text,
        new.file,
        new.page
    );
END;


-- 2) After DELETE on chunks => Remove the corresponding row from chunks_fts
CREATE TRIGGER IF NOT EXISTS chunks_ad
AFTER DELETE ON chunks
BEGIN
    INSERT INTO chunks_fts(
        chunks_fts,
        rowid,
        document_id,
        chunk_text,
        file,
        page
    )
    VALUES(
        'delete',
        old.id,
        old.document_id,
        old.chunk_text,
        old.file,
        old.page
    );
END;


-- 3) After UPDATE on chunks => Remove the old row, then insert the new row in chunks_fts
CREATE TRIGGER IF NOT EXISTS chunks_au
AFTER UPDATE ON chunks
BEGIN
    INSERT INTO chunks_fts(
        chunks_fts,
        rowid,
        document_id,
        chunk_text,
        file,
        page
    )
    VALUES(
        'delete',
        old.id,
        old.document_id,
        old.chunk_text,
        old.file,
        old.page
    );

    INSERT INTO chunks_fts(
        rowid,
        document_id,
        chunk_text,
        file,
        page
    )
    VALUES(
        new.id,
        new.document_id,
        new.chunk_text,
        new.file,
        new.page
    );
END;

