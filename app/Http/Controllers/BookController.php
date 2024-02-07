<?php

namespace App\Http\Controllers;

use App\Models\Book;
use Illuminate\Http\Request;

class BookController extends Controller
{
    public function welcome(){
        return view('welcome');
    }

    public function index(){

        $books = Book::limit(10)->get();


        return view('books.index')->with('books',$books);
    }
}
