-- CreateTable
CREATE TABLE "CompanyImage" (
    "id" SERIAL NOT NULL,
    "company_id" INTEGER NOT NULL,
    "company_logo" TEXT NOT NULL,

    CONSTRAINT "CompanyImage_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "CompanyTeams" (
    "id" SERIAL NOT NULL,
    "company_id" INTEGER NOT NULL,
    "title" TEXT NOT NULL,

    CONSTRAINT "CompanyTeams_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "CompanyLinks" (
    "id" SERIAL NOT NULL,
    "company_id" INTEGER NOT NULL,
    "url" TEXT NOT NULL,

    CONSTRAINT "CompanyLinks_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Company" (
    "id" SERIAL NOT NULL,
    "company_name" TEXT NOT NULL,
    "profile_description" TEXT,
    "location" TEXT NOT NULL,
    "company_webiste_url" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Company_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "JobPost" (
    "id" SERIAL NOT NULL,
    "job_post_url" TEXT NOT NULL,
    "slug" TEXT NOT NULL,
    "job_title" TEXT NOT NULL,
    "job_body" BYTEA NOT NULL,
    "company_code" INTEGER NOT NULL,
    "region" TEXT NOT NULL,
    "min_salary" INTEGER NOT NULL,
    "max_salary" INTEGER NOT NULL,
    "is_active" BOOLEAN NOT NULL,
    "is_remote" BOOLEAN NOT NULL,
    "is_sponsored" BOOLEAN NOT NULL,

    CONSTRAINT "JobPost_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "User" (
    "userId" SERIAL NOT NULL,
    "username" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "User_pkey" PRIMARY KEY ("userId")
);

-- CreateTable
CREATE TABLE "Tag" (
    "id" SERIAL NOT NULL,
    "tag_type" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL,
    "title" TEXT NOT NULL,

    CONSTRAINT "Tag_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "TagsOnPosts" (
    "postId" INTEGER NOT NULL,
    "tagId" INTEGER NOT NULL,
    "assignedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "assignedBy" TEXT NOT NULL,

    CONSTRAINT "TagsOnPosts_pkey" PRIMARY KEY ("postId","tagId")
);

-- CreateIndex
CREATE UNIQUE INDEX "CompanyImage_company_id_key" ON "CompanyImage"("company_id");

-- CreateIndex
CREATE UNIQUE INDEX "CompanyTeams_company_id_key" ON "CompanyTeams"("company_id");

-- CreateIndex
CREATE UNIQUE INDEX "CompanyLinks_company_id_key" ON "CompanyLinks"("company_id");

-- CreateIndex
CREATE UNIQUE INDEX "User_email_key" ON "User"("email");

-- AddForeignKey
ALTER TABLE "CompanyImage" ADD CONSTRAINT "CompanyImage_company_id_fkey" FOREIGN KEY ("company_id") REFERENCES "Company"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "CompanyTeams" ADD CONSTRAINT "CompanyTeams_company_id_fkey" FOREIGN KEY ("company_id") REFERENCES "Company"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "CompanyLinks" ADD CONSTRAINT "CompanyLinks_company_id_fkey" FOREIGN KEY ("company_id") REFERENCES "Company"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "JobPost" ADD CONSTRAINT "JobPost_company_code_fkey" FOREIGN KEY ("company_code") REFERENCES "Company"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "TagsOnPosts" ADD CONSTRAINT "TagsOnPosts_postId_fkey" FOREIGN KEY ("postId") REFERENCES "JobPost"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "TagsOnPosts" ADD CONSTRAINT "TagsOnPosts_tagId_fkey" FOREIGN KEY ("tagId") REFERENCES "Tag"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
