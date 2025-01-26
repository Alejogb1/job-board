import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { remark } from 'remark';
import html from 'remark-html';
type PostData = {
    id: string;
    date: string;
    title: string;
    contentHtml: string;
  };
  
  const gpuPostsDirectory = path.join(process.cwd(), 'gpu-blog-data');

export function getSortedGpuPostsData() {
    console.log("Fetching sorted GPU posts data...");
    const fileNames = fs.readdirSync(gpuPostsDirectory);
    console.log("GPU File names:", fileNames);
    
    const allGpuPostsData = fileNames.map((fileName) => {
      console.log(`Processing GPU file: ${fileName}`);
      const id = fileName.replace(/\.md$/, '');
      const fullPath = path.join(gpuPostsDirectory, fileName);
      const fileContents = fs.readFileSync(fullPath, 'utf8');
      const matterResult = matter(fileContents);
      console.log("Matter result:", matterResult);
      
      // Combine matter data with id, ensuring no duplicate properties
      return {
        ...matterResult.data as Omit<PostData, 'id'>,
        id,
      };
    });
    
    console.log("All GPU posts data before sorting:", allGpuPostsData);
    return allGpuPostsData.sort((a, b) => {
      if (a.date < b.date) {
        return 1;
      } else {
        return -1;
      }
    });
  }
  export async function getPostData(id: string) {
    console.log(`Fetching post data for ID: ${id}`);
    const fullPath = path.join(gpuPostsDirectory, `${id}.md`);
    const fileContents = fs.readFileSync(fullPath, 'utf8');
    const matterResult = matter(fileContents);
    
    console.log("Post metadata:", matterResult.data);
    
    const processedContent = await remark()
      .use(html)
      .process(matterResult.content);
    const contentHtml = processedContent.toString();
  
    return {
      id,
      contentHtml,
      ...matterResult.data,
    };
  }
  