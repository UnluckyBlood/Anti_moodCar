package com.example.automoodcar.network

import com.example.automoodcar.model.CarResponse
import retrofit2.http.GET
import retrofit2.http.Path

interface ApiService {
    @GET("car/{number}")
    suspend fun getCar(@Path("number") number: String): CarResponse
}