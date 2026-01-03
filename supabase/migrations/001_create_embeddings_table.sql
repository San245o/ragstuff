-- Enable the pgvector extension
create extension if not exists vector;

-- Create the document_chunks table with Gemini embedding dimensions (768)
create table if not exists document_chunks (
    id bigserial primary key,
    chunk_text text not null,
    chunk_index integer not null,
    page_number integer not null,
    start_char integer default 0,
    end_char integer default 0,
    embedding vector(768),  -- Gemini embedding dimension
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create index for faster similarity search using HNSW (better for 768 dims)
create index if not exists document_chunks_embedding_idx 
on document_chunks 
using hnsw (embedding vector_cosine_ops);

-- Create index for page_number lookups
create index if not exists document_chunks_page_number_idx 
on document_chunks (page_number);

-- Function to search similar chunks
create or replace function match_chunks (
  query_embedding vector(768),
  match_threshold float default 0.5,
  match_count int default 5
)
returns table (
  id bigint,
  chunk_text text,
  chunk_index integer,
  page_number integer,
  start_char integer,
  end_char integer,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    document_chunks.id,
    document_chunks.chunk_text,
    document_chunks.chunk_index,
    document_chunks.page_number,
    document_chunks.start_char,
    document_chunks.end_char,
    1 - (document_chunks.embedding <=> query_embedding) as similarity
  from document_chunks
  where 1 - (document_chunks.embedding <=> query_embedding) > match_threshold
  order by document_chunks.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Function to get chunk count
create or replace function get_chunk_count()
returns integer
language sql
as $$
  select count(*)::integer from document_chunks;
$$;
