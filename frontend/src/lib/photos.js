export const MAX_FILE_SIZE = 10 * 1024 * 1024

/** null when the file can be used, otherwise a plain-language reason. */
export function photoProblem(file) {
  if (!file) return 'Choose a photo first.'
  if (!file.type?.startsWith('image/')) return 'This file isn’t a photo. Choose a JPG, PNG or WebP.'
  if (file.size > MAX_FILE_SIZE) return 'This photo is larger than 10 MB. Choose a smaller one.'
  return null
}
