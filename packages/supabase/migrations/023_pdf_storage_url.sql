-- Add pdf_storage_url column to crawl_jobs for Supabase Storage references
ALTER TABLE crawl_jobs ADD COLUMN IF NOT EXISTS pdf_storage_url TEXT;

-- Create storage bucket for regulation PDFs (if not exists)
INSERT INTO storage.buckets (id, name, public)
VALUES ('regulation-pdfs', 'regulation-pdfs', true)
ON CONFLICT (id) DO NOTHING;

-- Public read policy for regulation PDFs
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'storage' AND tablename = 'objects'
      AND policyname = 'Public read regulation PDFs'
  ) THEN
    CREATE POLICY "Public read regulation PDFs"
      ON storage.objects FOR SELECT
      USING (bucket_id = 'regulation-pdfs');
  END IF;
END $$;
