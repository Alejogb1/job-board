export default function createSlug(companyName: string): string {
    // Remove leading and trailing spaces
    let slug = companyName.trim();
  
    // Replace spaces with hyphens
    slug = slug.replace(/\s+/g, '-');
  
    // Remove special characters and convert to lowercase
    slug = slug.replace(/[^a-zA-Z0-9-]/g, '').toLowerCase();
  
    return slug;
  }