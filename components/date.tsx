import { parseISO, format } from 'date-fns';

export default function Date({ dateString }: { dateString: string }) {
  // Check if dateString is defined and valid
  if (!dateString) {
    return <span>Invalid date</span>; // Handle undefined date
  }

  const date = parseISO(dateString); // Ensure dateString is valid ISO format
  return <time dateTime={dateString}>{date.toLocaleDateString()}</time>;
}