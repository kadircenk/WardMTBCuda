#include <iostream>
#include <vector>
#include <pthread.h>
#include "cuda_runtime.h"
#include "kernels.h"

#define THREAD_COUNT 16

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb-master/stb_image_write.h"

using namespace std;

void *calculateOffset(void *args)
{
	arguments *_arg;
	_arg = (arguments *)args;

	int width = _arg->width;
	int height = _arg->height;
	int first_index = _arg->first_ind;
	int second_index = _arg->second_ind;
	dim3 dimGrid, dimBlock;
	int tmpWidth = width / (pow(2, PYRAMID_LEVEL - 1));
	int tmpHeight = height / (pow(2, PYRAMID_LEVEL - 1));
	int tmpNImageSize = tmpWidth * tmpHeight;

	uint8_t **mtb = _arg->mtb;
	uint8_t **ebm = _arg->ebm;
	uint8_t **shifted_mtb = _arg->shifted_mtb;
	uint8_t **shifted_ebm = _arg->shifted_ebm;
	cudaStream_t stream = _arg->stream;

	int curr_level = PYRAMID_LEVEL - 1;
	int curr_offset_x = 0;
	int curr_offset_y = 0;
	int offset_return_x = 0;
	int offset_return_y = 0;

	for (int k = curr_level; k >= 0;
		 --k, tmpWidth *= 2, tmpHeight *= 2, tmpNImageSize *= 4)
	{
		curr_offset_x = 2 * offset_return_x;
		curr_offset_y = 2 * offset_return_y;

		int min_error = 255 * height * width;

		for (int i = -1; i <= 1; ++i)
		{
			for (int j = -1; j <= 1; ++j)
			{
				int xs = curr_offset_x + i;
				int ys = curr_offset_y + j;

				int x_shift = xs, y_shift = ys; //TODO check those

				int j_x, i_y, j_width, i_height;

				if (y_shift < 0)
				{ //height i
					i_y = -y_shift;
					i_height = tmpHeight;
				}
				else
				{
					i_y = 0;
					i_height = tmpHeight - y_shift;
				}

				if (x_shift < 0)
				{ //width j
					j_x = -x_shift;
					j_width = tmpWidth;
				}
				else
				{
					j_x = 0;
					j_width = tmpWidth - x_shift;
				}

				dimBlock = dim3(THREAD_COUNT, THREAD_COUNT);
				dimGrid = dim3(((j_width) + dimBlock.x - 1) / dimBlock.x,
							   ((i_height) + dimBlock.y - 1) / dimBlock.y);

				cudaMemset(shifted_mtb[second_index * PYRAMID_LEVEL + k], 0,
						   tmpNImageSize * sizeof(uint8_t));
				cudaMemset(shifted_ebm[second_index * PYRAMID_LEVEL + k], 0,
						   tmpNImageSize * sizeof(uint8_t));

				uint8_t *xor_result;
				cudaMalloc((void **)&xor_result,
						   tmpNImageSize * sizeof(uint8_t));

				uint8_t *after_ands;
				cudaMalloc((void **)&after_ands,
						   tmpNImageSize * sizeof(uint8_t));

				int *err;
				int error;
				cudaMalloc((void **)&err, sizeof(int));

				shift_Image<<<dimGrid, dimBlock, 0, stream>>>(
					shifted_mtb[second_index * PYRAMID_LEVEL + k],
					shifted_ebm[second_index * PYRAMID_LEVEL + k],
					mtb[second_index * PYRAMID_LEVEL + k],
					ebm[second_index * PYRAMID_LEVEL + k],
					tmpWidth,
					tmpHeight, xs, ys, j_x, i_y, j_width, i_height);

				dimBlock = dim3(THREAD_COUNT, THREAD_COUNT);
				dimGrid = dim3(((tmpWidth) + dimBlock.x - 1) / dimBlock.x,
							   ((tmpHeight) + dimBlock.y - 1) / dimBlock.y);

				XOR<<<dimGrid, dimBlock, 0, stream>>>(xor_result,
													  mtb[first_index * PYRAMID_LEVEL + k],
													  shifted_mtb[second_index * PYRAMID_LEVEL + k], tmpWidth,
													  tmpNImageSize);

				AND<<<dimGrid, dimBlock, 0, stream>>>(after_ands,
													  ebm[first_index * PYRAMID_LEVEL + k], shifted_ebm[second_index * PYRAMID_LEVEL + k],
													  xor_result,
													  tmpWidth, tmpNImageSize);

				count_Errors<<<32, 256, 0, stream>>>(after_ands, err,
													 tmpNImageSize);

				cudaMemcpyAsync(&error, err, sizeof(int), cudaMemcpyDeviceToHost, stream); //CANNOT BE ASYNC, SYNC PLEASE
				cudaStreamSynchronize(stream);

				if (error < min_error)
				{
					offset_return_x = xs;
					offset_return_y = ys;
					min_error = error;
				}
				//cudaFree(err);
			}
		}
	}

	_arg->shiftp->x = curr_offset_x;
	_arg->shiftp->y = curr_offset_y;

	pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
	if (argc == 1)
	{
		printf("Please supply the input images.");
		return 0;
	}

	int width, height;
	int img_count = argc - 1;
	int num_threads = img_count;
	pthread_t threads[num_threads];
	cudaStream_t streams[img_count];

	uint8_t *rgb_images[img_count];
	uint8_t *d_rgb_images[img_count];
	uint8_t *shifted_rgb_images[img_count];
	uint8_t *mtb[PYRAMID_LEVEL * img_count];
	uint8_t *ebm[PYRAMID_LEVEL * img_count];
	uint8_t *shifted_mtb[PYRAMID_LEVEL * img_count];
	uint8_t *shifted_ebm[PYRAMID_LEVEL * img_count];
	float *gray_image[PYRAMID_LEVEL * img_count];

	for (int i = 0; i < img_count; i++)
	{
		if (!(rgb_images[i] = stbi_load(argv[i + 1], &width, &height, NULL, 3)))
		{
			printf("Could not read image.");
			return 0;
		}
	}

	int nImageSize = width * height;				   //total pixel count of 1 channel of the RGB input image.
	size_t sizeOfImage = nImageSize * sizeof(uint8_t); //size of 1 channel of the RGB source image, each pixel is converted to uint8_t.

	//bu blok ustteki stbi_load ile 2 threadde ayni anda yapilir, start eventini bu yuzden altta baslatacagiz.
	for (int i = 0; i < img_count; i++)
	{
		int tmpSizeOfImage = sizeOfImage;
		int tmpWidth = width;
		int tmpHeight = height;
		int tmpNImageSize = nImageSize;

		for (int j = 0; j < PYRAMID_LEVEL; j++, tmpSizeOfImage /= 4, tmpWidth /= 2, tmpHeight /= 2, tmpNImageSize /= 4)
		{
			cudaMalloc((void **)&(gray_image[i * PYRAMID_LEVEL + j]),
					   tmpNImageSize * sizeof(float));
			cudaMalloc((void **)&(mtb[i * PYRAMID_LEVEL + j]),
					   tmpSizeOfImage);
			cudaMalloc((void **)&(ebm[i * PYRAMID_LEVEL + j]),
					   tmpSizeOfImage);
			cudaMalloc((void **)&(shifted_mtb[i * PYRAMID_LEVEL + j]),
					   tmpSizeOfImage);
			cudaMalloc((void **)&(shifted_ebm[i * PYRAMID_LEVEL + j]),
					   tmpSizeOfImage);
		}
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	for (int i = 0; i < img_count; i++)
	{
		cudaStreamCreate(&(streams[i]));
		cudaHostRegister(rgb_images[i], sizeOfImage * 3, 0);
		cudaMalloc((void **)&d_rgb_images[i], sizeOfImage * 3);
		cudaMemcpyAsync(d_rgb_images[i], rgb_images[i], sizeOfImage * 3,
						cudaMemcpyHostToDevice, streams[i]);
	}

	for (int i = 0; i < img_count; i++)
	{
		dim3 dimBlock = dim3(THREAD_COUNT, THREAD_COUNT);
		dim3 dimGrid = dim3((width + dimBlock.x - 1) / dimBlock.x,
							(height + dimBlock.y - 1) / dimBlock.y);

		convert_to_grayscale<<<dimGrid, dimBlock, 0, streams[i]>>>(
			gray_image[i * PYRAMID_LEVEL],
			d_rgb_images[i], nImageSize, width);
	}

	for (int i = 0; i < img_count; i++)
	{
		int tmpSizeOfImage = sizeOfImage;
		int tmpWidth = width;
		int tmpHeight = height;
		int tmpNImageSize = nImageSize;

		for (int j = 0; j < PYRAMID_LEVEL - 1; j++, tmpSizeOfImage /= 4, tmpWidth /= 2, tmpHeight /= 2, tmpNImageSize /= 4)
		{
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
			cudaArray *cuArray;
			cudaMallocArray(&cuArray, &channelDesc, tmpWidth,
							tmpHeight);

			cudaMemcpyToArray(cuArray, 0, 0,
							  gray_image[i * PYRAMID_LEVEL + j],
							  tmpNImageSize * sizeof(float),
							  cudaMemcpyDeviceToDevice);

			// Specify texture
			struct cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = cuArray;

			// Specify texture object parameters
			struct cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(texDesc));
			texDesc.addressMode[0] = cudaAddressModeClamp;
			texDesc.addressMode[1] = cudaAddressModeClamp;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode = cudaReadModeElementType;
			texDesc.normalizedCoords = 1;

			// Create texture object
			cudaTextureObject_t texObj = 0;
			cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

			// Invoke kernel
			dim3 dimBlock(THREAD_COUNT, THREAD_COUNT);
			dim3 dimGrid((tmpWidth / 2 + dimBlock.x - 1) / dimBlock.x,
						 (tmpHeight / 2 + dimBlock.y - 1) / dimBlock.y);

			transform_kernel<<<dimGrid, dimBlock, 0, streams[i]>>>(
				gray_image[i * PYRAMID_LEVEL + j + 1], texObj,
				tmpWidth / 2, tmpHeight / 2);
		}
	}

	for (int i = 0; i < img_count; i++)
	{
		int tmpSizeOfImage = sizeOfImage;
		int tmpWidth = width;
		int tmpHeight = height;
		int tmpNImageSize = nImageSize;

		for (int j = 0; j < PYRAMID_LEVEL; j++, tmpSizeOfImage /= 4, tmpWidth /= 2, tmpHeight /= 2, tmpNImageSize /= 4)
		{
			dim3 dimBlock(BLOCK_SIZE);
			dim3 dimGrid(32);

			int *hist;
			cudaMalloc((void **)&hist, BLOCK_SIZE * sizeof(int) * 32);

			histogram_smem_atomics<<<dimGrid, dimBlock, 0, streams[i]>>>(
				gray_image[i * PYRAMID_LEVEL + j], hist,
				tmpNImageSize);

			histogram_final_accum<<<1, 256, 0, streams[i]>>>(BLOCK_SIZE * dimGrid.x, hist);

			//TODO SUREKLI MEDIAN MALLOCLAMASIN, ARRAY ACALIM EN BASTAN
			//TODO BURALARI OPTIMIZE ET
			int *median;
			cudaMalloc((void **)&median, sizeof(int));
			find_Median<<<1, 1, 0, streams[i]>>>(tmpNImageSize, hist, median);

			// int *asd = (int *)malloc(
			// 			sizeof(int) * 1);
			// cudaMemcpy(asd, median,
			// 				sizeof(int) * 1, cudaMemcpyDeviceToHost);
			// cout<<"median: "<<*asd<<" ve n: "<<tmpNImageSize<<endl;

			dimBlock = dim3(THREAD_COUNT, THREAD_COUNT);
			dimGrid = dim3((tmpWidth + dimBlock.x - 1) / dimBlock.x,
						   (tmpHeight + dimBlock.y - 1) / dimBlock.y);

			find_Mtb_Ebm<<<dimGrid, dimBlock, 0, streams[i]>>>(
				gray_image[i * PYRAMID_LEVEL + j], median,
				mtb[i * PYRAMID_LEVEL + j],
				ebm[i * PYRAMID_LEVEL + j], tmpHeight, tmpWidth);
		}
	}

	// cudaDeviceSynchronize(); //wait for all streams to finish up.

	// char str[32];
	// sprintf(str, "mtb.jpg");
	// char path[80] = "./output/";
	// strcat(path, str);

	// uint8_t *tmpmtb = (uint8_t *)malloc(
	// 	sizeof(uint8_t) * nImageSize/1);
	// cudaMemcpy(tmpmtb, mtb[0],
	// 			sizeof(uint8_t) * nImageSize/1, cudaMemcpyDeviceToHost);

	// cudaDeviceSynchronize();

	// stbi_write_jpg(path, width/1, height/1, 1, tmpmtb, width*3);

	// return 0;

	// for (int var = 0; var < img_count; ++var)
	// {
	// 	cudaHostUnregister(rgb_images[var]);
	// }

	//**********************************************************************************************************************************
	shift_pair all_shifts[img_count];

	int mid_img_index = img_count / 2; //TODO belki +1.

	for (int m = mid_img_index - 1; m >= 0; --m)
	{
		arguments *args = (arguments *)malloc(sizeof(arguments));
		args->shiftp = &(all_shifts[m]);
		args->stream = streams[m];
		args->first_ind = m + 1;
		args->second_ind = m;
		args->width = width;
		args->height = height;
		args->mtb = mtb;
		args->ebm = ebm;
		args->shifted_mtb = shifted_mtb;
		args->shifted_ebm = shifted_ebm;

		//cout << "part1 thread create ediyoz..." << endl;
		int rc = pthread_create(&threads[m], NULL, calculateOffset,
								(void *)args);

		if (rc)
		{
			cout << "Error:unable to create thread," << rc << endl;
			exit(-1);
		}
	}
	for (int m = mid_img_index + 1; m < img_count; ++m)
	{

		arguments *args = (arguments *)malloc(sizeof(arguments));
		args->shiftp = &(all_shifts[m]);
		args->stream = streams[m];
		args->first_ind = m - 1;
		args->second_ind = m;
		args->width = width;
		args->height = height;
		args->mtb = mtb;
		args->ebm = ebm;
		args->shifted_mtb = shifted_mtb;
		args->shifted_ebm = shifted_ebm;

		//cout << "part2 thread create ediyoz..." << endl;
		int rc = pthread_create(&threads[m], NULL, calculateOffset,
								((void *)args));

		if (rc)
		{
			cout << "Error:unable to create thread," << rc << endl;
			exit(-1);
		}
	}

	//cudaDeviceSynchronize();

	void *status;
	for (int var = 0; var < num_threads; ++var)
	{
		if (var != mid_img_index)
			pthread_join(threads[var], &status);
	}
	//cout << "thread joinler bitti..." << endl;

	//cudaDeviceSynchronize();

	//cout << " ilk parttaki imajlari shiftliyoruz tek tek ..." << endl;

	int eskiTotalX = 0, eskiTotalY = 0;
	for (int m = mid_img_index - 1; m >= 0; --m)
	{

		int x_shift = all_shifts[m].x + eskiTotalX;
		int y_shift = all_shifts[m].y + eskiTotalY; //TODO check those

		int tmpWidth = width;
		int tmpHeight = height;
		int tmpNImageSize = tmpWidth * tmpHeight;

		//TODO bunu da bastn hallet
		cudaMalloc((void **)&shifted_rgb_images[m], 3 * tmpNImageSize);
		cudaMemset(shifted_rgb_images[m], 0,
				   3 * tmpNImageSize * sizeof(uint8_t));

		int j_x, i_y, j_width, i_height;

		if (y_shift < 0)
		{ //height i
			i_y = -y_shift;
			i_height = tmpHeight;
		}
		else
		{
			i_y = 0;
			i_height = tmpHeight - y_shift;
		}

		if (x_shift < 0)
		{ //width j
			j_x = -x_shift;
			j_width = tmpWidth;
		}
		else
		{
			j_x = 0;
			j_width = tmpWidth - x_shift;
		}

		dim3 dimBlock = dim3(THREAD_COUNT, THREAD_COUNT);
		dim3 dimGrid = dim3(((j_width) + dimBlock.x - 1) / dimBlock.x,
							((i_height) + dimBlock.y - 1) / dimBlock.y);

		RGB_shift_Image<<<dimGrid, dimBlock, 0, streams[m]>>>(
			shifted_rgb_images[m], d_rgb_images[m], tmpWidth, tmpHeight,
			x_shift, y_shift, j_x, i_y, j_width, i_height);

		eskiTotalX += all_shifts[m].x;
		eskiTotalY += all_shifts[m].y;
		//cout << "   shiftledik: x,y " << eskiTotalX << " " << eskiTotalY << endl;
	}

	eskiTotalX = 0;
	eskiTotalY = 0;
	for (int m = mid_img_index + 1; m < img_count; ++m)
	{

		int x_shift = all_shifts[m].x + eskiTotalX;
		int y_shift = all_shifts[m].y + eskiTotalY; //TODO check those

		int tmpWidth = width;
		int tmpHeight = height;
		int tmpNImageSize = tmpWidth * tmpHeight;

		//TODO bunu da bastn hallet
		cudaMalloc((void **)&shifted_rgb_images[m], 3 * tmpNImageSize);
		cudaMemset(shifted_rgb_images[m], 0,
				   3 * tmpNImageSize * sizeof(uint8_t));

		int j_x, i_y, j_width, i_height;

		if (y_shift < 0)
		{ //height i
			i_y = -y_shift;
			i_height = tmpHeight;
		}
		else
		{
			i_y = 0;
			i_height = tmpHeight - y_shift;
		}

		if (x_shift < 0)
		{ //width j
			j_x = -x_shift;
			j_width = tmpWidth;
		}
		else
		{
			j_x = 0;
			j_width = tmpWidth - x_shift;
		}

		dim3 dimBlock = dim3(THREAD_COUNT, THREAD_COUNT);
		dim3 dimGrid = dim3(((j_width) + dimBlock.x - 1) / dimBlock.x,
							((i_height) + dimBlock.y - 1) / dimBlock.y);

		RGB_shift_Image<<<dimGrid, dimBlock, 0, streams[m]>>>(
			shifted_rgb_images[m], d_rgb_images[m], tmpWidth, tmpHeight,
			x_shift, y_shift, j_x, i_y, j_width, i_height);

		eskiTotalX += all_shifts[m].x;
		eskiTotalY += all_shifts[m].y;
		//cout << "   shiftledik: x,y " << eskiTotalX << " " << eskiTotalY << endl;
	}

	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "Time took: " << milliseconds << " ms" << endl;

	//print original grayscale img
	//	stbi_write_png(path, width, height, 3, rgb_images[mid_img_index], width*3);

	int tmpNImageSize = height * width;
	int tmpHeight = height;
	int tmpWidth = width;

	for (int var = 0; var < img_count; ++var)
	{

		if (var == mid_img_index)
		{
			continue;
		}

		char str[12];
		sprintf(str, "%d%d%d.jpg", var + 1, var + 1, var + 1);
		char path[80] = "./output/";
		strcat(path, str);

		uint8_t *tmpmtb = (uint8_t *)malloc(
			sizeof(uint8_t) * tmpNImageSize * 3);
		cudaMemcpy(tmpmtb, shifted_rgb_images[var],
				   sizeof(uint8_t) * tmpNImageSize * 3, cudaMemcpyDeviceToHost);

		stbi_write_jpg(path, tmpWidth, tmpHeight, 3, tmpmtb, tmpWidth * 3);
	}
	char str[12];
	sprintf(str, "%d%d%d.jpg", mid_img_index + 1, mid_img_index + 1, mid_img_index + 1);
	char path[80] = "./output/";
	strcat(path, str);

	uint8_t *tmpmtb = (uint8_t *)malloc(sizeof(uint8_t) * tmpNImageSize * 3);
	cudaMemcpy(tmpmtb, d_rgb_images[mid_img_index],
			   sizeof(uint8_t) * tmpNImageSize * 3, cudaMemcpyDeviceToHost);

	stbi_write_jpg(path, tmpWidth, tmpHeight, 3, tmpmtb, tmpWidth * 3);

	printf("Done. Wrote output images to output folder.\n");

	return 0;
}
