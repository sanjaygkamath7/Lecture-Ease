# Use official Node.js image
FROM node:18

# Set working directory
WORKDIR /app

# Copy only package files first
COPY package*.json ./

# Install ALL dependencies (including react-scripts in devDependencies)
RUN npm install

# Copy the rest of the app
COPY . .

# Expose React app port
EXPOSE 3000

# Start the React app
CMD ["npm", "start"]
