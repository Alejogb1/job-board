import { z } from 'zod'

const ACCEPTED_IMAGE_TYPES = [
    "image/jpeg",
    "image/jpg",
    "pdf",
    "docx",
  ]

export const FormDataSchema = z.object({
  firstName: z.string().min(1, 'First name is required'),
  lastName: z.string().min(1, 'Last name is required'),
  email: z.string().min(1, 'Email is required').email('Invalid email address'),
  country: z.string().min(1, 'Country is required'),
  street: z.string().min(1, 'Street is required'),
  city: z.string().min(1, 'City is required'),
  state: z.string().min(1, 'State is required'),
  zip: z.string().min(1, 'Zip is required'),
  file: z.object({
    name: z.string().refine((value) => value.toLowerCase().endsWith('.pdf'), {
      message: 'Invalid file format. Please upload a PDF file.',
    }),
    size: z.number().max(5 * 1024 * 1024, {
      message: 'File size exceeds the allowed limit of 5MB.',
    }),
    type: z.string().refine((value) => value === 'application/pdf', {
      message: 'Invalid file type. Please upload a PDF file.',
    }),
  }),
});
