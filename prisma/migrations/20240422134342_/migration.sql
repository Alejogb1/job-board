/*
  Warnings:

  - You are about to drop the column `job_post_url` on the `JobPost` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "JobPost" DROP COLUMN "job_post_url",
ALTER COLUMN "id" DROP DEFAULT,
ALTER COLUMN "job_body" DROP NOT NULL;
DROP SEQUENCE "JobPost_id_seq";
